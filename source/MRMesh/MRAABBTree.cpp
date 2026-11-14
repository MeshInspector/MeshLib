#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRPrefetch.h"
#include "MRRegionBoundary.h"
#include "MRAABBTreeCenters.hpp"
#include "MRTriBox.hpp"
#include <algorithm>
#include <tbb/parallel_invoke.h>

namespace MR
{

namespace
{

/// the box of a triangle of AABBTree, micro expanded to avoid leaks due to float errors
Box3f wideTriBox( const Mesh & mesh, const AABBTree::LeafTri & lt )
{
    return insignificantlyExpandedTriBox( mesh.points[lt.verts[0]], mesh.points[lt.verts[1]], mesh.points[lt.verts[2]] );
}

/// a triangle with its vertices and box, both found in one pass over the topology
struct BoxedTri
{
    AABBTree::LeafTri tri;
    Box3f box;

    explicit BoxedTri( NoInit ) noexcept : tri{ FaceId( noInit ), { VertId( noInit ), VertId( noInit ), VertId( noInit ) } }, box( noInit ) { }
};

/// the numFaces triangles with their boxes, the doubled centers of these boxes, and the box of the centers
Buffer<BoxedTri> makeBoxedTris( const MeshPart & mp, int numFaces, Buffer<TreeCenter> & centers, Box3f & centersBox )
{
    Buffer<BoxedTri> boxedTris( numFaces );
    centers = Buffer<TreeCenter>( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            boxedTris[n++].tri.face = f;
    }

    const auto & topology = mp.mesh.topology;
    const auto & points = mp.mesh.points;
    const auto faceOf = [&] ( int i ) { return packed ? FaceId( i ) : boxedTris[i].tri.face; };
    // the records of a triangle are often far from the previous triangle's in memory, so they are read ahead:
    // its first edge's 2 * cAhead triangles ahead, its next edge's cAhead triangles ahead, and its points cAhead / 4 triangles ahead
    constexpr int cAhead = 16;
    constexpr int cBlock = 1024; // the centers are included in their box by blocks, while in the cache
    centersBox = tbb::parallel_reduce( tbb::blocked_range<int>( 0, numFaces ), Box3f(),
        [&] ( const tbb::blocked_range<int> & range, Box3f box )
    {
        const int end = range.end();
        for ( int b = range.begin(); b < end; b += cBlock )
        {
            for ( int i = b, be = std::min( end, b + cBlock ); i < be; ++i )
            {
                if ( i + 2 * cAhead < end )
                    topology.prefetchEdge( topology.edgeWithLeft( faceOf( i + 2 * cAhead ) ) );
                if ( i + cAhead < end )
                    topology.prefetchEdge( topology.prev( topology.edgeWithLeft( faceOf( i + cAhead ) ).sym() ) );
                if ( i + cAhead / 4 < end )
                    for ( auto v : topology.getTriVerts( faceOf( i + cAhead / 4 ) ) )
                        MR_PREFETCH( &points[v] );
                auto & bt = boxedTris[i];
                const FaceId f = faceOf( i );
                bt.tri = { f, topology.getTriVerts( f ) };
                insignificantlyExpandedTriBoxAndCenter( points[bt.tri.verts[0]], points[bt.tri.verts[1]], points[bt.tri.verts[2]], i, bt.box, centers[i] );
            }
            box = includeCenters( box, centers.data() + b, std::min( end, b + cBlock ) - b );
        }
        return box;
    },
        [] ( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    return boxedTris;
}

/// the triangles of t (all if region is nullptr) with their boxes, see makeBoxedTris above
Buffer<BoxedTri> makeBoxedTris( const VertCoords & points, const Triangulation & t, const FaceBitSet * region, int numFaces,
    Buffer<TreeCenter> & centers, Box3f & centersBox )
{
    Buffer<BoxedTri> boxedTris( numFaces );
    centers = Buffer<TreeCenter>( numFaces );
    const bool packed = numFaces == (int)t.size();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : *region )
            boxedTris[n++].tri.face = f;
    }

    const auto faceOf = [&] ( int i ) { return packed ? FaceId( i ) : boxedTris[i].tri.face; };
    constexpr int cAhead = 4; // the points of a triangle are read cAhead triangles ahead
    constexpr int cBlock = 1024; // the centers are included in their box by blocks, while in the cache
    centersBox = tbb::parallel_reduce( tbb::blocked_range<int>( 0, numFaces ), Box3f(),
        [&] ( const tbb::blocked_range<int> & range, Box3f box )
    {
        const int end = range.end();
        for ( int b = range.begin(); b < end; b += cBlock )
        {
            for ( int i = b, be = std::min( end, b + cBlock ); i < be; ++i )
            {
                if ( i + cAhead < end )
                    for ( auto v : t[faceOf( i + cAhead )] )
                        MR_PREFETCH( &points[v] );
                const FaceId f = faceOf( i );
                const ThreeVertIds v = t[f];
                // not bt.tri = { f, v }, which MSVC assembles on the stack and copies by one wider load, waiting for the stores
                auto & bt = boxedTris[i];
                bt.tri.face = f;
                bt.tri.verts = v;
                insignificantlyExpandedTriBoxAndCenter( points[v[0]], points[v[1]], points[v[2]], i, bt.box, centers[i] );
            }
            box = includeCenters( box, centers.data() + b, std::min( end, b + cBlock ) - b );
        }
        return box;
    },
        [] ( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    return boxedTris;
}

/// splits each AABBTree node in two binary levels, each at the median of the triangle box centers
/// along the largest extent of these centers; so the nodes at binary depth d have ( numTris >> d ) or ( numTris >> d ) + 1 triangles
class AABBTreeWideMaker
{
public:
    using Node = AABBTree::Node;

    /// counts the nodes of the tree of numTris triangles
    explicit AABBTreeWideMaker( int numTris ) : numTris_( numTris )
    {
        for ( auto & c : cnt_ )
            c = { -1, -1 };
        totalNodes_ = countNodes_( 0, numTris_ );
    }

    [[nodiscard]] int numNodes() const { return totalNodes_; }

    /// fills nodes, and the triangles and their boxes in leaf order (all of the sizes for numTris)
    void make( const Buffer<BoxedTri> & boxedTris, Buffer<TreeCenter> & centers, const Box3f & centersBox,
        Node * nodes, AABBTree::LeafTri * leafTris, Box3f * leafBoxes )
    {
        assert( (int)boxedTris.size() == numTris_ && (int)centers.size() == numTris_ );
        boxedTris_ = boxedTris.data();
        centers_ = centers.data();
        nodes_ = nodes;
        leafTris_ = leafTris;
        leafBoxes_ = leafBoxes;
        parallelSplits_ = numTris_ >= cParallelSplitNum
            && tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) > 1
            && tbb::this_task_arena::max_concurrency() > 1;
        if ( numTris_ >= cBucketSplitNum )
            scratch_ = Buffer<TreeCenter>( numTris_ );
        make_( NodeId( 0 ), NodeId( 1 ), { 0, { 0, numTris_, false, true, centersBox } } );
    }

private:
    /// centers [first, first + num) either in centers_ or at the same positions in scratch_, and the box of them if known
    struct Range
    {
        int first, num;
        bool inScratch;
        bool hasBox = false;
        Box3f box;
    };
    struct Child
    {
        int depth;
        Range r;
    };

    TreeCenter * data_( const Range & r ) const { return ( r.inScratch ? scratch_.data() : centers_ ) + r.first; }

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

    Box3f centersBox_( const Range & r ) const
    {
        const TreeCenter * d = data_( r );
        if ( !parallelSplits_ || r.num < cParallelSplitNum )
            return includeCenters( Box3f(), d, r.num );
        return tbb::parallel_reduce( tbb::blocked_range<int>( 0, r.num, 1 << 14 ), Box3f(),
            [d]( const tbb::blocked_range<int> & br, Box3f b ) { return includeCenters( b, d + br.begin(), (int)br.size() ); },
            []( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    }

    // [first, first + num / 2) get the smaller centers along the largest extent of the centers (ties are ordered by index); returns the two halves
    std::array<Range, 2> split_( const Range & r )
    {
        const auto box = r.hasBox ? r.box : centersBox_( r );
        const auto sz = box.size();
        const int axis = sz.x >= sz.y ? ( sz.x >= sz.z ? 0 : 2 ) : ( sz.y >= sz.z ? 1 : 2 );
        TreeCenter * const data = data_( r );
        if ( r.num >= cBucketSplitNum )
        {
            const int chunkSize = parallelSplits_ && r.num >= cParallelSplitNum ? cChunk : r.num;
            Box3f halfBoxes[2];
            if ( splitCentersByBuckets( data, data_( { r.first, r.num, !r.inScratch, false, {} } ), r.num, axis, box.min[axis], box.max[axis], chunkSize, halfBoxes ) )
                return { Range{ r.first, r.num / 2, !r.inScratch, true, halfBoxes[0] }, Range{ r.first + r.num / 2, r.num - r.num / 2, !r.inScratch, true, halfBoxes[1] } };
        }
        selectCenter( data, data + r.num / 2, data + r.num, axis );
        return { Range{ r.first, r.num / 2, r.inScratch, false, {} }, Range{ r.first + r.num / 2, r.num - r.num / 2, r.inScratch, false, {} } };
    }

    // fills node n with its subtree, whose nodes below the children of n start from desc
    void make_( NodeId n, NodeId desc, Child c )
    {
        auto & node = nodes_[int( n )];
        const int first = c.r.first, num = c.r.num;
        if ( num <= AABBTree::MaxLeafTris )
        {
            // the triangles of a leaf in the order of their indices
            static_assert( AABBTree::MaxLeafTris <= 4, "the sorting network below is for up to 4 triangles" );
            TreeCenter * const leaf = data_( c.r );
            const auto cs = [leaf]( int x, int y ) { if ( leaf[y].i < leaf[x].i ) std::swap( leaf[x], leaf[y] ); };
            switch ( num )
            {
            case 2: cs( 0, 1 ); break;
            case 3: cs( 0, 1 ); cs( 1, 2 ); cs( 0, 1 ); break;
            case 4: cs( 0, 1 ); cs( 2, 3 ); cs( 0, 2 ); cs( 1, 3 ); cs( 1, 2 ); break;
            default: break;
            }
            Box3f box;
            for ( int i = 0; i < num; ++i )
            {
                const auto & bt = boxedTris_[leaf[i].i];
                box.include( bt.box );
                leafTris_[first + i] = bt.tri;
                leafBoxes_[first + i] = bt.box;
            }
            node.box = box;
            node.first = first;
            node.count = -num;
            return;
        }

        const auto halves = split_( c.r );
        std::array<Range, 2> quarters[2];
        const bool parallel = num >= 4096;
        if ( parallel )
            tbb::parallel_invoke( [&] { quarters[0] = split_( halves[0] ); }, [&] { quarters[1] = split_( halves[1] ); } );
        Child ch[AABBTree::MaxChildren];
        int k = 0;
        for ( int j = 0; j < 2; ++j )
        {
            if ( halves[j].num <= AABBTree::MaxLeafTris )
            {
                ch[k++] = { c.depth + 1, halves[j] };
                continue;
            }
            if ( !parallel )
                quarters[j] = split_( halves[j] );
            ch[k++] = { c.depth + 2, quarters[j][0] };
            ch[k++] = { c.depth + 2, quarters[j][1] };
        }
        node.first = int( desc );
        node.count = k;

        NodeId childDesc[AABBTree::MaxChildren];
        NodeId next = desc + k;
        for ( int j = 0; j < k; ++j )
        {
            childDesc[j] = next;
            next = next + ( numNodes_( ch[j].depth, ch[j].r.num ) - 1 );
        }
        if ( !parallel )
        {
            for ( int j = 0; j < k; ++j )
                make_( desc + j, childDesc[j], ch[j] );
        }
        else
        {
            tbb::task_group group;
            for ( int j = 1; j < k; ++j )
                group.run( [&, j] { make_( desc + j, childDesc[j], ch[j] ); } );
            make_( desc, childDesc[0], ch[0] );
            group.wait();
        }

        Box3f box;
        for ( int j = 0; j < k; ++j )
            box.include( nodes_[int( desc ) + j].box );
        node.box = box;
    }

    const BoxedTri * boxedTris_ = nullptr;
    TreeCenter * centers_ = nullptr;
    Buffer<TreeCenter> scratch_; // the centers of a range move between centers_ and scratch_ by each split of the range by buckets
    static constexpr int cBucketSplitNum = 1024; // smaller ranges are split by selectCenter
    static constexpr int cParallelSplitNum = 1 << 17;
    static constexpr int cChunk = 1 << 14; // the centers counted and moved by one thread in a parallel split
    bool parallelSplits_ = false; // on one thread the parallel split only adds passes over the range
    Node * nodes_ = nullptr;
    AABBTree::LeafTri * leafTris_ = nullptr;
    Box3f * leafBoxes_ = nullptr;
    int numTris_ = 0;
    int totalNodes_ = 0;
    std::array<int, 2> cnt_[64]; // by binary depth and the number of triangles minus ( numTris_ >> depth )
};

/// makes the tree of numTris triangles given by makeBoxedTris( centers, centersBox )
template <typename MakeBoxedTris>
void makeTree( int numTris, const MakeBoxedTris & makeBoxedTris,
    Vector<AABBTree::Node, NodeId> & nodes, Vector<AABBTree::LeafTri> & leafTris, Vector<Box3f> & leafBoxes )
{
    if ( numTris <= 0 )
        return;
    AABBTreeWideMaker maker( numTris );
    Buffer<TreeCenter> centers;
    Box3f centersBox;
    Buffer<BoxedTri> boxedTris;
    // the outputs are filled with their default values (by one thread) while the triangles are boxed
    tbb::parallel_invoke( [&] { boxedTris = makeBoxedTris( centers, centersBox ); },
        [&] { nodes.resize( maker.numNodes() ); leafTris.resize( numTris ); leafBoxes.resizeNoInit( numTris ); } );
    maker.make( boxedTris, centers, centersBox, nodes.data(), leafTris.data(), leafBoxes.data() );
}

} // anonymous namespace

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;
    const auto numTris = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    makeTree( numTris, [&] ( Buffer<TreeCenter> & centers, Box3f & centersBox ) { return makeBoxedTris( mp, numTris, centers, centersBox ); },
        nodes_, leafTris_, leafBoxes_ );
}

AABBTree::AABBTree( const VertCoords & points, const Triangulation & t, const FaceBitSet * region )
{
    MR_TIMER;
    const auto numTris = region ? (int)region->count() : (int)t.size();
    makeTree( numTris, [&] ( Buffer<TreeCenter> & centers, Box3f & centersBox ) { return makeBoxedTris( points, t, region, numTris, centers, centersBox ); },
        nodes_, leafTris_, leafBoxes_ );
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
