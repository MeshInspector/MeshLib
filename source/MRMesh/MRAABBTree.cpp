#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include <algorithm>
#include <tbb/parallel_invoke.h>

namespace MR
{

namespace
{

/// the box of a triangle of AABBTree, micro expanded to avoid leaks due to float errors
Box3f wideTriBox( const Mesh & mesh, const AABBTree::LeafTri & lt )
{
    Box3f box;
    for ( auto v : lt.verts )
        box.include( mesh.points[v] );
    return box.insignificantlyExpanded();
}

/// a triangle with its vertices and box, both found in one pass over the topology
struct BoxedTri
{
    AABBTree::LeafTri tri;
    Box3f box;

    explicit BoxedTri( NoInit ) noexcept : tri{ FaceId( noInit ), { VertId( noInit ), VertId( noInit ), VertId( noInit ) } }, box( noInit ) { }
};

Buffer<BoxedTri> makeBoxedTris( const MeshPart & mp )
{
    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return {};

    Buffer<BoxedTri> boxedTris( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            boxedTris[n++].tri.face = f;
    }

    ParallelFor( 0, numFaces, [&] ( int i )
    {
        auto & bt = boxedTris[i];
        const FaceId f = packed ? FaceId( i ) : bt.tri.face;
        bt.tri = { f, mp.mesh.topology.getTriVerts( f ) };
        bt.box = wideTriBox( mp.mesh, bt.tri );
    } );
    return boxedTris;
}

/// splits each AABBTree node in two binary levels, each at the median of the triangle box centers
/// along the largest extent of these centers; so the nodes at binary depth d have ( numTris >> d ) or ( numTris >> d ) + 1 triangles
class AABBTreeWideMaker
{
public:
    using Node = AABBTree::Node;

    /// sorted instead of BoxedTri: smaller and compared by one coordinate
    struct Center
    {
        int i; ///< in boxedTris
        Vector3f c; ///< doubled center of the box

        Center( int i, const Vector3f & c ) noexcept : i( i ), c( c ) { }
        explicit Center( NoInit ) noexcept : c( noInit ) { }
    };

    /// fills nodes, and the triangles and their boxes in leaf order
    AABBTreeWideMaker( const Buffer<BoxedTri> & boxedTris, Buffer<Center> & centers,
        Vector<Node, NodeId> & nodes, Vector<AABBTree::LeafTri> & leafTris, Vector<Box3f> & leafBoxes )
        : boxedTris_( boxedTris ), centers_( centers ), nodes_( nodes ), leafTris_( leafTris ), leafBoxes_( leafBoxes )
    {
        numTris_ = (int)boxedTris_.size();
        for ( auto & c : cnt_ )
            c = { -1, -1 };
        nodes_.resize( countNodes_( 0, numTris_ ) );
        parallelSplits_ = numTris_ >= cParallelSplitNum
            && tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) > 1
            && tbb::this_task_arena::max_concurrency() > 1;
        if ( parallelSplits_ )
            scratch_ = Buffer<Center>( numTris_ );
        make_( NodeId( 0 ), NodeId( 1 ), 0, 0, numTris_ );
    }

private:
    struct Child
    {
        int depth, first, num;
    };

    // the number of nodes in the subtree of a node at binary depth d with num triangles
    int countNodes_( int d, int num )
    {
        int & res = cnt_[d][num - ( numTris_ >> d )];
        if ( res >= 0 )
            return res;
        if ( num <= AABBTree::MaxLeafTris )
            return res = 1;
        res = 1;
        const int halves[2] = { num / 2, num - num / 2 };
        for ( int h : halves )
        {
            if ( h <= AABBTree::MaxLeafTris )
                res += countNodes_( d + 1, h );
            else
                res += countNodes_( d + 2, h / 2 ) + countNodes_( d + 2, h - h / 2 );
        }
        return res;
    }
    int numNodes_( int d, int num ) const { return cnt_[d][num - ( numTris_ >> d )]; }

    int largestCentersAxis_( int first, int num ) const
    {
        Box3f box;
        if ( parallelSplits_ && num >= cParallelSplitNum )
        {
            box = tbb::parallel_reduce( tbb::blocked_range<int>( first, first + num, 1 << 14 ), Box3f(),
                [&]( const tbb::blocked_range<int> & r, Box3f b )
                {
                    for ( int i = r.begin(); i < r.end(); ++i )
                        b.include( centers_[i].c );
                    return b;
                },
                []( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
        }
        else
        {
            for ( int i = 0; i < num; ++i )
                box.include( centers_[first + i].c );
        }
        const auto sz = box.size();
        return sz.x >= sz.y ? ( sz.x >= sz.z ? 0 : 2 ) : ( sz.y >= sz.z ? 1 : 2 );
    }

    // [first, first + num / 2) get the smaller centers along axis; ties are ordered by index to make the tree independent of std::nth_element
    void split_( int first, int num, int axis )
    {
        const auto less = [axis]( const Center & a, const Center & b ) { return a.c[axis] < b.c[axis] || ( a.c[axis] == b.c[axis] && a.i < b.i ); };
        Center * const data = centers_.data() + first;
        const int mid = num / 2;
        if ( !parallelSplits_ || num < cParallelSplitNum )
        {
            std::nth_element( data, data + mid, data + num, less );
            return;
        }

        // a big range is partitioned in parallel in three parts by two samples around the median,
        // and only the small middle part with the median is given to std::nth_element
        constexpr int cNumSamples = 1024, cHalfGap = 64;
        Buffer<Center> samples( cNumSamples );
        for ( int k = 0; k < cNumSamples; ++k )
            samples[k] = data[ std::int64_t( k ) * num / cNumSamples ];
        std::sort( samples.data(), samples.data() + cNumSamples, less );
        const Center lo = samples[cNumSamples / 2 - cHalfGap];
        const Center hi = samples[cNumSamples / 2 + cHalfGap];
        const auto part = [&]( const Center & c ) { return less( c, lo ) ? 0 : ( less( c, hi ) ? 1 : 2 ); };

        constexpr int cChunk = 1 << 13;
        const int numChunks = ( num + cChunk - 1 ) / cChunk;
        struct Counts { int n[3]; };
        std::vector<Counts> counts( numChunks );
        ParallelFor( 0, numChunks, [&]( int ch )
        {
            Counts cnt{};
            for ( int i = ch * cChunk, e = std::min( num, i + cChunk ); i < e; ++i )
                ++cnt.n[part( data[i] )];
            counts[ch] = cnt;
        } );
        Counts total{};
        for ( const auto & c : counts )
            for ( int k = 0; k < 3; ++k )
                total.n[k] += c.n[k];
        if ( mid < total.n[0] || mid >= total.n[0] + total.n[1] )
        {
            // the median is not between the samples
            std::nth_element( data, data + mid, data + num, less );
            return;
        }
        Counts pos{ { 0, total.n[0], total.n[0] + total.n[1] } };
        for ( auto & c : counts )
        {
            const auto my = pos;
            for ( int k = 0; k < 3; ++k )
                pos.n[k] += c.n[k];
            c = my;
        }
        Center * const tmp = scratch_.data() + first;
        ParallelFor( 0, numChunks, [&]( int ch )
        {
            auto pos = counts[ch];
            for ( int i = ch * cChunk, e = std::min( num, i + cChunk ); i < e; ++i )
                tmp[ pos.n[part( data[i] )]++ ] = data[i];
        } );
        ParallelFor( 0, numChunks, [&]( int ch )
        {
            const int i = ch * cChunk, e = std::min( num, i + cChunk );
            std::copy( tmp + i, tmp + e, data + i );
        } );
        std::nth_element( data + total.n[0], data + mid, data + total.n[0] + total.n[1], less );
    }

    // fills node n with its subtree, whose nodes below the children of n start from desc
    void make_( NodeId n, NodeId desc, int depth, int first, int num )
    {
        auto & node = nodes_[n];
        if ( num <= AABBTree::MaxLeafTris )
        {
            // nth_element leaves the triangles of a half in an order of its implementation
            std::sort( centers_.data() + first, centers_.data() + first + num, []( const Center & a, const Center & b ) { return a.i < b.i; } );
            Box3f box;
            for ( int i = first; i < first + num; ++i )
            {
                const auto & bt = boxedTris_[centers_[i].i];
                box.include( bt.box );
                leafTris_[i] = bt.tri;
                leafBoxes_[i] = bt.box;
            }
            node.box = box;
            node.first = first;
            node.count = -num;
            return;
        }

        const int axis = largestCentersAxis_( first, num );
        split_( first, num, axis );
        const Child halves[2] = { { depth + 1, first, num / 2 }, { depth + 1, first + num / 2, num - num / 2 } };
        Child ch[AABBTree::MaxChildren];
        int k = 0;
        const auto splitHalf = [&]( const Child & h ) { split_( h.first, h.num, largestCentersAxis_( h.first, h.num ) ); };
        const bool parallel = num >= 4096;
        if ( parallel )
            tbb::parallel_invoke( [&] { splitHalf( halves[0] ); }, [&] { splitHalf( halves[1] ); } );
        for ( const auto & h : halves )
        {
            if ( h.num <= AABBTree::MaxLeafTris )
            {
                ch[k++] = h;
                continue;
            }
            if ( !parallel )
                splitHalf( h );
            ch[k++] = { depth + 2, h.first, h.num / 2 };
            ch[k++] = { depth + 2, h.first + h.num / 2, h.num - h.num / 2 };
        }
        node.first = int( desc );
        node.count = k;

        NodeId childDesc[AABBTree::MaxChildren];
        NodeId next = desc + k;
        for ( int j = 0; j < k; ++j )
        {
            childDesc[j] = next;
            next = next + ( numNodes_( ch[j].depth, ch[j].num ) - 1 );
        }
        if ( !parallel )
        {
            for ( int j = 0; j < k; ++j )
                make_( desc + j, childDesc[j], ch[j].depth, ch[j].first, ch[j].num );
        }
        else
        {
            tbb::task_group group;
            for ( int j = 1; j < k; ++j )
                group.run( [&, j] { make_( desc + j, childDesc[j], ch[j].depth, ch[j].first, ch[j].num ); } );
            make_( desc, childDesc[0], ch[0].depth, ch[0].first, ch[0].num );
            group.wait();
        }

        Box3f box;
        for ( int j = 0; j < k; ++j )
            box.include( nodes_[desc + j].box );
        node.box = box;
    }

    const Buffer<BoxedTri> & boxedTris_;
    Buffer<Center> & centers_;
    Buffer<Center> scratch_; // for parallel splits of big ranges
    static constexpr int cParallelSplitNum = 1 << 17;
    bool parallelSplits_ = false; // on one thread the parallel split only adds passes over the range
    Vector<Node, NodeId> & nodes_;
    Vector<AABBTree::LeafTri> & leafTris_;
    Vector<Box3f> & leafBoxes_;
    int numTris_ = 0;
    std::array<int, 2> cnt_[64]; // by binary depth and the number of triangles minus ( numTris_ >> depth )
};

} // anonymous namespace

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;
    auto boxedTris = makeBoxedTris( mp );
    if ( boxedTris.size() == 0 )
        return;
    Buffer<AABBTreeWideMaker::Center> centers( boxedTris.size() );
    ParallelFor( size_t( 0 ), centers.size(), [&]( size_t i )
    {
        const auto & box = boxedTris[i].box;
        centers[i] = AABBTreeWideMaker::Center( int( i ), box.min + box.max );
    } );
    leafTris_.resize( boxedTris.size() );
    leafBoxes_.resizeNoInit( boxedTris.size() );
    AABBTreeWideMaker maker( boxedTris, centers, nodes_, leafTris_, leafBoxes_ );
}

std::vector<NodeId> AABBTree::getSubtrees( int minNum ) const
{
    MR_TIMER;
    assert( minNum > 0 );
    std::vector<NodeId> res;
    if ( nodes_.empty() )
        return res;
    res.push_back( NodeId( 0 ) );
    std::vector<NodeId> tmp;
    while ( res.size() < minNum )
    {
        bool split = false;
        for ( NodeId n : res )
        {
            const auto & node = nodes_[n];
            if ( node.leaf() )
            {
                tmp.push_back( n );
                continue;
            }
            for ( int i = 0; i < node.count; ++i )
                tmp.push_back( NodeId( node.first + i ) );
            split = true;
        }
        res.swap( tmp );
        tmp.clear();
        if ( !split )
            break;
    }
    return res;
}

FaceBitSet AABBTree::getSubtreeFaces( NodeId subtreeRoot ) const
{
    MR_TIMER;
    FaceBitSet res;
    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( subtreeRoot );
    while ( !subtasks.empty() )
    {
        const auto & node = nodes_[subtasks.top()];
        subtasks.pop();
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                res.autoResizeSet( leafTris_[i].face );
            continue;
        }
        for ( int i = node.count - 1; i >= 0; --i )
            subtasks.push( NodeId( node.first + i ) );
    }
    return res;
}

NodeBitSet AABBTree::getNodesFromFaces( const FaceBitSet & faces ) const
{
    MR_TIMER;
    NodeBitSet res( nodes_.size() );
    if ( nodes_.empty() )
        return res;
    BitSetParallelForAll( res, [&]( NodeId n )
    {
        const auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            if ( faces.test( leafTris_[i].face ) )
            {
                res.set( n );
                return;
            }
        }
    } );
    // children follow their parent
    for ( auto n = nodes_.backId(); n.valid(); --n )
    {
        const auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        for ( int i = 0; i < node.count; ++i )
        {
            if ( res.test( NodeId( node.first + i ) ) )
            {
                res.set( n );
                break;
            }
        }
    }
    return res;
}

void AABBTree::getLeafOrder( FaceBMap & faceMap ) const
{
    MR_TIMER;
    for ( size_t i = 0; i < leafTris_.size(); ++i )
        faceMap.b[leafTris_[i].face] = FaceId( i );
    faceMap.tsize = decltype( faceMap.tsize )( leafTris_.size() );
}

void AABBTree::renumber( const FaceBMap & faceMap, const VertBMap & vertMap )
{
    MR_TIMER;
    ParallelFor( leafTris_, [&]( size_t i )
    {
        auto & lt = leafTris_[i];
        lt.face = faceMap.b[lt.face];
        for ( auto & v : lt.verts )
            v = vertMap.b[v];
    } );
}

void AABBTree::flipOrientation( const FaceBitSet * region )
{
    MR_TIMER;
    ParallelFor( leafTris_, [&]( size_t i )
    {
        auto & lt = leafTris_[i];
        if ( !region || region->test( lt.face ) )
            std::swap( lt.verts[0], lt.verts[1] );
    } );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( nodes_.empty() )
        return;
    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );

    // the boxes of the leaves with changed triangles
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId n )
    {
        auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        bool changed = false;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            if ( changedVerts && !changedFaces.test( leafTris_[i].face ) )
                continue;
            leafBoxes_[i] = wideTriBox( mesh, leafTris_[i] );
            changed = true;
        }
        if ( !changed )
            return;
        changedNodes.set( n );
        Box3f box;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
            box.include( leafBoxes_[i] );
        node.box = box;
    } );

    // the other nodes from the last: children follow their parent
    for ( auto n = nodes_.backId(); n.valid(); --n )
    {
        auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        bool changed = false;
        for ( int i = 0; i < node.count && !changed; ++i )
            changed = changedNodes.test( NodeId( node.first + i ) );
        if ( !changed )
            continue;
        changedNodes.set( n );
        Box3f box;
        for ( int i = 0; i < node.count; ++i )
            box.include( nodes_[NodeId( node.first + i )].box );
        node.box = box;
    }
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !nodes_.empty() );

    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        splitFaces.set( oldFace );

    // the leaf of every split face, and the new vertices and boxes of the split faces
    std::vector<FaceId> oldFaces; // in increasing order
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    std::vector<NodeId> leafOfOld( oldFaces.size() );
    ParallelFor( nodes_, [&]( NodeId n )
    {
        const auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            auto & lt = leafTris_[i];
            if ( !splitFaces.test( lt.face ) )
                continue;
            lt.verts = mesh.topology.getTriVerts( lt.face );
            leafBoxes_[i] = wideTriBox( mesh, lt );
            const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), lt.face );
            assert( it != oldFaces.end() && *it == lt.face );
            leafOfOld[it - oldFaces.begin()] = n;
        }
    } );

    // the new faces grouped by the leaf of their old face, in a deterministic order
    std::vector<std::pair<NodeId, FaceId>> leafNew;
    leafNew.reserve( new2Old.size() );
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        const auto leaf = leafOfOld[std::lower_bound( oldFaces.begin(), oldFaces.end(), oldFace ) - oldFaces.begin()];
        if ( !leaf )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        leafNew.emplace_back( leaf, newFace );
    }
    std::sort( leafNew.begin(), leafNew.end(), []( const auto & x, const auto & y ) { return std::tie( x.first, x.second ) < std::tie( y.first, y.second ); } );

    // every changed leaf becomes a node with the old leaf as the first child and the leaves of the new faces;
    // the new nodes are appended, so children have larger ids than their parents as in a constructed tree
    const auto firstNewNode = nodes_.endId();
    std::vector<NodeId> changedLeaves;
    // appends a block of up to MaxChildren nodes for the given children, every child is a leaf or a list of more than MaxChildren leaves
    auto makeChildren = [&]( NodeId parent, std::vector<Node> leaves, auto && self ) -> void
    {
        const int n = (int)leaves.size();
        const int numCh = std::min( n, MaxChildren );
        const NodeId first = nodes_.endId();
        nodes_[parent].first = int( first );
        nodes_[parent].count = numCh;
        nodes_.resize( nodes_.size() + numCh );
        if ( n <= MaxChildren )
        {
            for ( int i = 0; i < n; ++i )
                nodes_[first + i] = leaves[i];
            return;
        }
        for ( int c = 0; c < numCh; ++c )
        {
            std::vector<Node> part( leaves.begin() + c * n / numCh, leaves.begin() + ( c + 1 ) * n / numCh );
            if ( part.size() == 1 )
                nodes_[first + c] = part.front();
            else
                self( first + c, std::move( part ), self );
        }
    };
    for ( size_t k = 0; k < leafNew.size(); )
    {
        const auto leaf = leafNew[k].first;
        std::vector<Node> leaves{ nodes_[leaf] }; // the old leaf keeps its triangles
        for ( ; k < leafNew.size() && leafNew[k].first == leaf; )
        {
            Node nl;
            nl.first = (int)leafTris_.size();
            int num = 0;
            for ( ; num < MaxLeafTris && k < leafNew.size() && leafNew[k].first == leaf; ++num, ++k )
            {
                const auto f = leafNew[k].second;
                leafTris_.push_back( { f, mesh.topology.getTriVerts( f ) } );
                leafBoxes_.push_back( wideTriBox( mesh, leafTris_.back() ) );
            }
            nl.count = -num;
            leaves.push_back( nl );
        }
        makeChildren( leaf, std::move( leaves ), makeChildren );
        changedLeaves.push_back( leaf );
    }

    // the boxes of the new nodes from the last, and then of the changed leaves (now parents)
    auto updateBox = [&]( NodeId n )
    {
        auto & node = nodes_[n];
        Box3f box;
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                box.include( leafBoxes_[i] );
        }
        else
        {
            for ( int i = 0; i < node.count; ++i )
                box.include( nodes_[NodeId( node.first + i )].box );
        }
        node.box = box;
    };
    for ( auto n = nodes_.backId(); n >= firstNewNode; --n )
        updateBox( n );
    bool grown = false; // whether the subtree of some split face is not inside the box of its old leaf
    for ( auto n : changedLeaves )
    {
        const auto oldBox = nodes_[n].box;
        updateBox( n );
        grown = grown || !oldBox.contains( nodes_[n].box );
    }

    // only a new vertex outside the box of its split face (e.g. due to rounding) makes the boxes of the ancestors to grow,
    // then update all not-leaf nodes from the last to the root as in refit()
    if ( grown )
        for ( auto n = nodes_.backId(); n.valid(); --n )
            if ( !nodes_[n].leaf() )
                updateBox( n );

    // every split leaf went one level deeper: the stacks of the traversals hold the nodes of MaxDepth levels only
    std::vector<int> depth( nodes_.size(), 0 );
    int maxDepth = 0;
    for ( NodeId n( 0 ); n < nodes_.size(); ++n )
    {
        const auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        for ( int i = 0; i < node.count; ++i )
            maxDepth = std::max( maxDepth, depth[node.first + i] = depth[n] + 1 );
    }
    if ( maxDepth > MaxDepth )
    {
        FaceBitSet faces( mesh.topology.faceSize() );
        for ( const auto & lt : leafTris_ )
            faces.set( lt.face );
        *this = AABBTree( MeshPart{ mesh, &faces } );
    }
}

} //namespace MR
