#include "MRAABBTree.h"
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

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;

    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return;

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

    // (split face, its new part) sorted for the subtrees not to depend on the order of the hash map
    std::vector<std::pair<FaceId, FaceId>> splits;
    splits.reserve( new2Old.size() );
    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        splits.emplace_back( oldFace, newFace );
        splitFaces.set( oldFace );
    }
    std::sort( splits.begin(), splits.end() );

    // find the leaf of every split face
    std::vector<FaceId> oldFaces;
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    std::vector<NodeId> oldLeaves( oldFaces.size() );
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        const auto & node = nodes_[nid];
        if ( !node.leaf() || !splitFaces.test( node.leafId() ) )
            return;
        const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), node.leafId() );
        assert( it != oldFaces.end() && *it == node.leafId() );
        oldLeaves[it - oldFaces.begin()] = nid;
    } );

    // the subtree of the parts: its root takes the place of the old leaf, other nodes are appended,
    // so children have larger ids than their parents as in a constructed tree
    nodes_.reserve( nodes_.size() + 2 * splits.size() );
    std::vector<BoxedFace> parts;
    auto addPart = [&]( FaceId f )
    {
        auto & part = parts.emplace_back();
        part.leafId = f;
        part.box = computeFaceBox( mesh, f );
    };
    auto makeSubtree = [&]( auto && self, NodeId nid, int first, int num ) -> void
    {
        if ( num == 1 )
        {
            nodes_[nid].setLeafId( parts[first].leafId );
            nodes_[nid].box = parts[first].box;
            return;
        }
        const auto l = nodes_.endId();
        const auto r = l + 1;
        nodes_.resize( nodes_.size() + 2 );
        const int numL = num / 2;
        self( self, l, first, numL );
        self( self, r, first + numL, num - numL );
        auto & node = nodes_[nid];
        node.l = l;
        node.r = r;
        node.box = nodes_[l].box;
        node.box.include( nodes_[r].box );
    };

    std::vector<NodeId> grownRoots; // whose subtree is not inside the box of the old leaf
    size_t i = 0;
    for ( size_t j = 0; j < oldFaces.size(); ++j )
    {
        const auto oldFace = oldFaces[j];
        parts.clear();
        addPart( oldFace );
        for ( ; i < splits.size() && splits[i].first == oldFace; ++i )
            addPart( splits[i].second );
        const auto root = oldLeaves[j];
        if ( !root )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        const auto oldBox = nodes_[root].box;
        makeSubtree( makeSubtree, root, 0, int( parts.size() ) );
        if ( !oldBox.contains( nodes_[root].box ) )
            grownRoots.push_back( root );
    }
    assert( i == splits.size() );

    // only a new vertex outside the box of its split face makes the boxes of the ancestors to grow
    if ( grownRoots.empty() )
        return;
    Vector<NodeId, NodeId> parents( nodes_.size() );
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        const auto & node = nodes_[nid];
        if ( node.leaf() )
            return;
        parents[node.l] = nid;
        parents[node.r] = nid;
    } );
    for ( auto root : grownRoots )
    {
        const auto box = nodes_[root].box;
        for ( auto p = parents[root]; p; p = parents[p] )
        {
            if ( nodes_[p].box.contains( box ) )
                break; // and all further ancestors contain it too
            nodes_[p].box.include( box );
        }
    }
}

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
