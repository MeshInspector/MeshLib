#include "MRAABBTree.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRResizeNoInit.h"
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

    Leaves boxedFaces;
    resizeNoInit( boxedFaces, numFaces );
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

    build_( std::move( boxedFaces ) );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;

    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );
    auto changedLeaf = [&]( NodeId c )
    {
        return isLeaf( c ) && ( !changedVerts || changedFaces.test( leafId( c ) ) );
    };

    // update leaf boxes
    ParallelFor( leaves_, [&]( size_t i )
    {
        auto & l = leaves_[i];
        if ( !changedVerts || changedFaces.test( l.leafId ) )
            l.box = computeFaceBox( mesh, l.leafId );
    } );

    // update not-leaf nodes
    NodeBitSet changedNodes( nodes_.size() );
    auto changed = [&]( NodeId c )
    {
        return isLeaf( c ) ? changedLeaf( c ) : changedNodes.test( c );
    };
    for ( auto nid = NodeId( int( nodes_.size() ) - 1 ); nid >= 0; --nid )
    {
        auto & node = nodes_[nid];
        if ( !changed( node.l ) && !changed( node.r ) )
            continue;
        changedNodes.set( nid );
        node.box = box( node.l );
        node.box.include( box( node.r ) );
    }
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !empty() );

    // the index of the leaf of every split face
    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        splitFaces.set( oldFace );
    std::vector<FaceId> oldFaces; // in increasing order
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    auto oldFaceIndex = [&]( FaceId f )
    {
        const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), f );
        assert( it != oldFaces.end() && *it == f );
        return it - oldFaces.begin();
    };
    constexpr size_t cNotFound = ~size_t( 0 );
    std::vector<size_t> leafIndices( oldFaces.size(), cNotFound );
    ParallelFor( leaves_, [&]( size_t i )
    {
        if ( splitFaces.test( leaves_[i].leafId ) )
            leafIndices[oldFaceIndex( leaves_[i].leafId )] = i;
    } );

    // the place referencing the leaf: in the node (as its left or right child) or the root if node is invalid
    struct Place
    {
        NodeId node;
        bool right = false;
    };
    std::vector<Place> places( oldFaces.size() );
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        for ( bool right : { false, true } )
        {
            const auto c = right ? nodes_[nid].r : nodes_[nid].l;
            if ( isLeaf( c ) && splitFaces.test( leafId( c ) ) )
                places[oldFaceIndex( leafId( c ) )] = { nid, right };
        }
    } );
    NodeId root; // only for a single-leaf tree, which is split
    auto ref = [&]( Place p ) -> NodeId &
    {
        return !p.node ? root : p.right ? nodes_[p.node].r : nodes_[p.node].l;
    };

    // every new face turns the current leaf of its split face into a node with the leaves of both faces;
    // the new nodes and leaves are appended, so children have larger ids than their parents as in a constructed tree
    std::vector<BoxT> oldBoxes( oldFaces.size() );
    for ( size_t i = 0; i < oldFaces.size(); ++i )
    {
        if ( leafIndices[i] == cNotFound )
            continue;
        auto & l = leaves_[leafIndices[i]];
        oldBoxes[i] = l.box;
        l.box = computeFaceBox( mesh, l.leafId );
    }
    const auto firstNewNode = nodes_.endId();
    nodes_.reserve( nodes_.size() + new2Old.size() );
    leaves_.reserve( leaves_.size() + new2Old.size() );
    auto places1 = places; // current places of the leaves of split faces
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        const auto i = oldFaceIndex( oldFace );
        if ( leafIndices[i] == cNotFound )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        const auto nid = nodes_.endId();
        auto & node = nodes_.emplace_back();
        node.l = Node::leafChild( leafIndices[i] );
        node.r = Node::leafChild( leaves_.size() );
        auto & leaf = leaves_.emplace_back();
        leaf.leafId = newFace;
        leaf.box = computeFaceBox( mesh, newFace );
        ref( places1[i] ) = nid;
        places1[i] = { nid, false };
    }
    for ( auto n = nodes_.backId(); n >= firstNewNode; --n )
    {
        auto & node = nodes_[n];
        node.box = box( node.l );
        node.box.include( box( node.r ) );
    }

    // only a new vertex outside the box of its split face (e.g. due to rounding) makes the boxes of the ancestors to grow,
    // then update all old not-leaf nodes from the last to the root as in refit()
    bool grown = false;
    for ( size_t i = 0; i < oldFaces.size() && !grown; ++i )
        if ( leafIndices[i] != cNotFound )
            grown = !oldBoxes[i].contains( box( ref( places[i] ) ) );
    if ( grown )
    {
        for ( auto n = firstNewNode - 1; n >= 0; --n )
        {
            auto & node = nodes_[n];
            node.box = box( node.l );
            node.box.include( box( node.r ) );
        }
    }
}

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
