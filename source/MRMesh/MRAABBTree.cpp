#include "MRAABBTree.h"
#include "MRAABBTreeWide.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include <algorithm>

namespace MR
{

using BoxedFace = BoxedLeaf<FaceTreeTraits3>;

inline Box3f computeFaceBox( const Mesh & mesh, FaceId f )
{
    Box3f box;
    Vector3f a, b, c;
    mesh.getTriPoints( f, a, b, c );
    box.include( a );
    box.include( b );
    box.include( c );

    // micro expand boxes to have better precision in AABB algorithms
    // insignificantlyExpanded - needed to avoid leaks due to float errors
    // (small intersection of neighbor boxes guarantee that both of them will be considered as candidates of connection area)
    box = box.insignificantlyExpanded();
    return box;
}

static Buffer<BoxedFace> makeBoxedFaces( const MeshPart & mp )
{
    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return {};

    Buffer<BoxedFace> boxedFaces( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            boxedFaces[n++].leafId = f;
    }

    // compute aabb's of each face
    ParallelFor( 0, numFaces, [&] ( int i )
    {
        FaceId f;
        if ( packed )
            boxedFaces[i].leafId = f = FaceId( i );
        else
            f = boxedFaces[i].leafId;
        boxedFaces[i].box = computeFaceBox( mp.mesh, f );
    } );
    return boxedFaces;
}

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;
    auto boxedFaces = makeBoxedFaces( mp );
    if ( boxedFaces.size() == 0 )
        return;
    nodes_ = makeAABBTreeNodeVec( std::move( boxedFaces ) );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;

    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );

    // update leaf nodes
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( !node.leaf() )
            return;
        const auto f = node.leafId();
        if ( changedVerts && !changedFaces.test( f ) )
            return;
        changedNodes.set( nid );
        node.box = computeFaceBox( mesh, f );
    } );

    //update not-leaf nodes
    for ( auto nid = nodes_.backId(); nid; --nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            continue;
        if ( !changedNodes.test( node.l ) && !changedNodes.test( node.r ) )
            continue;
        changedNodes.set( nid );
        node.box = nodes_[node.l].box;
        node.box.include( nodes_[node.r].box );
    }
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !nodes_.empty() );

    // find the leaf of every split face
    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        splitFaces.set( oldFace );
    std::vector<FaceId> oldFaces; // in increasing order
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    std::vector<NodeId> roots( oldFaces.size() ); // the leaves of split faces become the roots of the subtrees of their parts
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        const auto & node = nodes_[nid];
        if ( !node.leaf() || !splitFaces.test( node.leafId() ) )
            return;
        const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), node.leafId() );
        assert( it != oldFaces.end() && *it == node.leafId() );
        roots[it - oldFaces.begin()] = nid;
    } );

    // every new face turns the current leaf of its split face into a node with the leaves of both faces;
    // the new nodes are appended, so children have larger ids than their parents as in a constructed tree
    const auto firstNewNode = nodes_.endId();
    nodes_.reserve( nodes_.size() + 2 * new2Old.size() );
    auto leaves = roots; // current leaves of split faces
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        auto & leaf = leaves[std::lower_bound( oldFaces.begin(), oldFaces.end(), oldFace ) - oldFaces.begin()];
        if ( !leaf )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        const auto l = nodes_.endId();
        const auto r = l + 1;
        nodes_.resize( nodes_.size() + 2 );
        nodes_[l].setLeafId( oldFace );
        nodes_[r].setLeafId( newFace );
        nodes_[leaf].l = l;
        nodes_[leaf].r = r;
        leaf = l;
    }

    // the boxes of the new nodes from the leaves, and then of the roots
    auto updateBox = [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            node.box = computeFaceBox( mesh, node.leafId() );
        else
        {
            node.box = nodes_[node.l].box;
            node.box.include( nodes_[node.r].box );
        }
    };
    for ( auto nid = nodes_.backId(); nid >= firstNewNode; --nid )
        updateBox( nid );
    bool grown = false; // whether the subtree of some split face is not inside the box of its old leaf
    for ( auto root : roots )
    {
        if ( !root )
            continue;
        const auto oldBox = nodes_[root].box; // still the box of the old leaf
        updateBox( root );
        grown = grown || !oldBox.contains( nodes_[root].box );
    }

    // only a new vertex outside the box of its split face (e.g. due to rounding) makes the boxes of the ancestors to grow,
    // then update all not-leaf nodes from the last to the root as in refit()
    if ( grown )
        for ( auto nid = nodes_.backId(); nid; --nid )
            if ( !nodes_[nid].leaf() )
                updateBox( nid );
}

namespace
{

/// splits AABBTreeWide nodes by the medians of their triangles as AABBTree does, two binary levels per node;
/// so the nodes at binary depth d have ( numTris >> d ) or ( numTris >> d ) + 1 triangles
class AABBTreeWideMaker
{
public:
    using Node = AABBTreeWide::Node;
    AABBTreeWideMaker( Buffer<BoxedFace> & boxedFaces, Vector<Node, NodeId> & nodes ) : boxedFaces_( boxedFaces ), nodes_( nodes )
    {
        numTris_ = (int)boxedFaces_.size();
        for ( auto & c : cnt_ )
            c = { -1, -1 };
        nodes_.resize( countNodes_( 0, numTris_ ) );
        make_( NodeId( 0 ), NodeId( 1 ), 0, 0, numTris_ );
    }

private:
    struct Child
    {
        int depth, first, num;
    };

    // children of a node with numTris > MaxLeafTris: the halves, and the halves of the halves with more than MaxLeafTris
    int children_( int depth, int first, int num, const Box3f & box, Child (&res)[AABBTreeWide::MaxChildren] ) const
    {
        const int mid = partitionBoxedLeaves<FaceTreeTraits3>( boxedFaces_.data(), box, first, first + num );
        const Child halves[2] = { { depth + 1, first, mid - first }, { depth + 1, mid, first + num - mid } };
        int k = 0;
        for ( const auto & h : halves )
        {
            if ( h.num <= AABBTreeWide::MaxLeafTris )
            {
                res[k++] = h;
                continue;
            }
            const int hmid = partitionBoxedLeaves<FaceTreeTraits3>( boxedFaces_.data(), box_( h.first, h.num ), h.first, h.first + h.num );
            res[k++] = { depth + 2, h.first, hmid - h.first };
            res[k++] = { depth + 2, hmid, h.first + h.num - hmid };
        }
        return k;
    }

    // the number of nodes in the subtree of a node at binary depth d with num triangles
    int countNodes_( int d, int num )
    {
        int & res = cnt_[d][num - ( numTris_ >> d )];
        if ( res >= 0 )
            return res;
        if ( num <= AABBTreeWide::MaxLeafTris )
            return res = 1;
        res = 1;
        const int halves[2] = { num / 2, num - num / 2 };
        for ( int h : halves )
        {
            if ( h <= AABBTreeWide::MaxLeafTris )
                res += countNodes_( d + 1, h );
            else
                res += countNodes_( d + 2, h / 2 ) + countNodes_( d + 2, h - h / 2 );
        }
        return res;
    }
    int numNodes_( int d, int num ) const { return cnt_[d][num - ( numTris_ >> d )]; }

    Box3f box_( int first, int num ) const
    {
        Box3f box;
        for ( int i = 0; i < num; ++i )
            box.include( boxedFaces_[first + i].box );
        return box;
    }

    // fills node n with its subtree, whose nodes below the children of n start from desc
    void make_( NodeId n, NodeId desc, int depth, int first, int num )
    {
        auto & node = nodes_[n];
        node.box = box_( first, num );
        if ( num <= AABBTreeWide::MaxLeafTris )
        {
            node.first = first;
            node.count = -num;
            return;
        }
        Child ch[AABBTreeWide::MaxChildren];
        const int k = children_( depth, first, num, node.box, ch );
        node.first = int( desc );
        node.count = k;
        NodeId childDesc[AABBTreeWide::MaxChildren];
        NodeId next = desc + k;
        for ( int j = 0; j < k; ++j )
        {
            childDesc[j] = next;
            next = next + ( numNodes_( ch[j].depth, ch[j].num ) - 1 );
        }
        if ( num < 4096 )
        {
            for ( int j = 0; j < k; ++j )
                make_( desc + j, childDesc[j], ch[j].depth, ch[j].first, ch[j].num );
            return;
        }
        tbb::task_group group;
        for ( int j = 1; j < k; ++j )
            group.run( [&, j] { make_( desc + j, childDesc[j], ch[j].depth, ch[j].first, ch[j].num ); } );
        make_( desc, childDesc[0], ch[0].depth, ch[0].first, ch[0].num );
        group.wait();
    }

    Buffer<BoxedFace> & boxedFaces_;
    Vector<Node, NodeId> & nodes_;
    int numTris_ = 0;
    std::array<int, 2> cnt_[64]; // by binary depth and the number of triangles minus ( numTris_ >> depth )
};

} // anonymous namespace

AABBTreeWide::AABBTreeWide( const MeshPart & mp )
{
    MR_TIMER;
    auto boxedFaces = makeBoxedFaces( mp );
    if ( boxedFaces.size() == 0 )
        return;
    AABBTreeWideMaker maker( boxedFaces, nodes_ );
    leafTris_.resize( boxedFaces.size() );
    ParallelFor( size_t( 0 ), leafTris_.size(), [&]( size_t i )
    {
        const auto f = boxedFaces[i].leafId;
        leafTris_[i] = { f, mp.mesh.topology.getTriVerts( f ) };
    } );
}

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
