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

    build_( std::move( boxedFaces ), mp.mesh.topology.faceSize() );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( empty() )
        return;

    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );
    auto changedLeaf = [&]( NodeId c )
    {
        return isLeaf( c ) && ( !changedVerts || changedFaces.test( leafId( c ) ) );
    };

    // update leaf boxes, and the boxes of the nodes with changed leaf children only from these children
    auto updateLeaf = [&]( NodeId c )
    {
        if ( changedLeaf( c ) )
            leafBoxes_[leafId( c )] = computeFaceBox( mesh, leafId( c ) );
    };
    updateLeaf( root_ );
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( !changedLeaf( node.l ) && !changedLeaf( node.r ) )
            return;
        changedNodes.set( nid );
        node.box = {};
        for ( auto c : { node.l, node.r } )
        {
            if ( !isLeaf( c ) )
                continue;
            updateLeaf( c );
            node.box.include( leafBoxes_[leafId( c )] );
        }
    } );

    // update not-leaf nodes from not-leaf children
    for ( auto nid = NodeId( int( nodes_.size() ) - 1 ); nid >= 0; --nid )
    {
        auto & node = nodes_[nid];
        const bool lChanged = !isLeaf( node.l ) && changedNodes.test( node.l );
        const bool rChanged = !isLeaf( node.r ) && changedNodes.test( node.r );
        if ( changedNodes.test( nid ) )
        {
            // the box already includes leaf children
            if ( !isLeaf( node.l ) )
                node.box.include( nodes_[node.l].box );
            if ( !isLeaf( node.r ) )
                node.box.include( nodes_[node.r].box );
        }
        else if ( lChanged || rChanged )
        {
            changedNodes.set( nid );
            node.box = box( node.l );
            node.box.include( box( node.r ) );
        }
    }
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !empty() );

    // find the place referencing the leaf of every split face
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

    // the place of a leaf: in the node (as its left or right child) or in root_ if node is invalid
    struct Place
    {
        NodeId node;
        bool right = false;
    };
    std::vector<Place> places( oldFaces.size() );
    std::vector<char> found( oldFaces.size(), false );
    auto findLeaf = [&]( NodeId c, Place p )
    {
        if ( !isLeaf( c ) || !splitFaces.test( leafId( c ) ) )
            return;
        const auto i = oldFaceIndex( leafId( c ) );
        places[i] = p;
        found[i] = true;
    };
    findLeaf( root_, {} );
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        findLeaf( nodes_[nid].l, { nid, false } );
        findLeaf( nodes_[nid].r, { nid, true } );
    } );
    auto ref = [&]( Place p ) -> NodeId &
    {
        return !p.node ? root_ : p.right ? nodes_[p.node].r : nodes_[p.node].l;
    };

    // every new face turns the current leaf of its split face into a node with the leaves of both faces;
    // the new nodes are appended, so children have larger ids than their parents as in a constructed tree
    const auto firstNewNode = nodes_.endId();
    nodes_.resize( nodes_.size() + new2Old.size() );
    auto nid = firstNewNode;
    auto leaves = places; // current places of the leaves of split faces
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        const auto i = oldFaceIndex( oldFace );
        if ( !found[i] )
        {
            assert( false ); // the split face is not in this tree
            nodes_.pop_back();
            continue;
        }
        auto & node = nodes_[nid];
        node.l = Node::leafChild( oldFace );
        node.r = Node::leafChild( newFace );
        ref( leaves[i] ) = nid;
        leaves[i] = { nid, false };
        ++nid;
    }

    // the boxes of all leaves of split faces, then of the new nodes from the leaves
    std::vector<BoxT> oldBoxes( oldFaces.size() );
    for ( size_t i = 0; i < oldFaces.size(); ++i )
        if ( found[i] )
            oldBoxes[i] = leafBoxes_[oldFaces[i]];
    leafBoxes_.resize( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        leafBoxes_[newFace] = computeFaceBox( mesh, newFace );
    for ( size_t i = 0; i < oldFaces.size(); ++i )
        if ( found[i] )
            leafBoxes_[oldFaces[i]] = computeFaceBox( mesh, oldFaces[i] );
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
        if ( found[i] )
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
