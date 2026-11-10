#include "MRAABBTree.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRResizeNoInit.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRBuffer.h"
#include <algorithm>

namespace MR
{

using BoxedFace = BoxedLeaf<FaceTreeTraits3>;

inline Box3f computeTriBox( const VertCoords & points, const ThreeVertIds & v )
{
    Box3f box( points[v[0]], points[v[0]] );
    box.include( points[v[1]] );
    box.include( points[v[2]] );

    // micro expand boxes to have better precision in AABB algorithms
    // insignificantlyExpanded - needed to avoid leaks due to float errors
    // (small intersection of neighbor boxes guarantee that both of them will be considered as candidates of connection area)
    return box.insignificantlyExpanded();
}

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;

    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return;

    // the i-th boxed face and tris address the i-th face of the region
    Vector<BoxedFace> boxedFaces;
    boxedFaces.resizeNoInit( numFaces );
    Buffer<VertId> tris( 3 * size_t( numFaces ) );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    Buffer<FaceId> faces;
    if ( !packed )
    {
        faces = Buffer<FaceId>( numFaces );
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            faces[n++] = f;
    }

    ParallelFor( 0, numFaces, [&] ( int i )
    {
        const auto v = mp.mesh.topology.getTriVerts( packed ? FaceId( i ) : faces[i] );
        for ( int j = 0; j < 3; ++j )
            tris[3 * size_t( i ) + j] = v[j];
        boxedFaces[i].box = computeTriBox( mp.mesh.points, v );
    } );

    leaves_.resizeNoInit( numFaces );
    leafBoxes_.resizeNoInit( numFaces );
    nodes_ = makeAABBTreeNodeVec( boxedFaces, [&]( size_t pos, size_t i )
    {
        leafBoxes_[pos] = boxedFaces[i].box;
        leaves_[pos].leafId = packed ? FaceId( i ) : faces[i];
        for ( int j = 0; j < 3; ++j )
            leaves_[pos].verts[j] = tris[3 * i + j];
    } );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( nodes_.empty() )
    {
        if ( !leaves_.empty() )
            leafBoxes_[0] = computeTriBox( mesh.points, leaves_[0].verts );
        return;
    }

    if ( !changedVerts )
    {
        // the boxes of leaves and of leaf children in parallel, then the boxes of not-leaf children from the last node to the root
        ParallelFor( leafBoxes_, [&]( size_t i )
        {
            leafBoxes_[i] = computeTriBox( mesh.points, leaves_[i].verts );
        } );
        ParallelFor( nodes_, [&]( NodeId nid )
        {
            auto & node = nodes_[nid];
            node.box = {};
            if ( isLeaf( node.l ) )
                node.box.include( leafBox( node.l ) );
            if ( isLeaf( node.r ) )
                node.box.include( leafBox( node.r ) );
        } );
        for ( auto nid = nodes_.backId(); nid >= 0; --nid )
        {
            auto & node = nodes_[nid];
            if ( !isLeaf( node.l ) )
                node.box.include( nodes_[node.l].box );
            if ( !isLeaf( node.r ) )
                node.box.include( nodes_[node.r].box );
        }
        return;
    }

    const auto changedFaces = getIncidentFaces( mesh.topology, *changedVerts );
    ParallelFor( leafBoxes_, [&]( size_t i )
    {
        if ( changedFaces.test( leaves_[i].leafId ) )
            leafBoxes_[i] = computeTriBox( mesh.points, leaves_[i].verts );
    } );
    NodeBitSet changedNodes( nodes_.size() );
    auto changed = [&]( NodeId c )
    {
        return isLeaf( c ) ? changedFaces.test( leafId( c ) ) : changedNodes.test( c );
    };
    for ( auto nid = nodes_.backId(); nid >= 0; --nid )
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
    Vector<size_t> leafIndices( oldFaces.size(), cNotFound );
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
    Vector<Place> places( oldFaces.size() );
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
    Vector<BoxT> oldBoxes( oldFaces.size() );
    for ( size_t i = 0; i < oldFaces.size(); ++i )
    {
        if ( leafIndices[i] == cNotFound )
            continue;
        oldBoxes[i] = leafBoxes_[leafIndices[i]];
        auto & leaf = leaves_[leafIndices[i]];
        leaf.verts = mesh.topology.getTriVerts( oldFaces[i] );
        leafBoxes_[leafIndices[i]] = computeTriBox( mesh.points, leaf.verts );
    }
    const auto firstNewNode = nodes_.endId();
    nodes_.reserve( nodes_.size() + new2Old.size() );
    leaves_.reserve( leaves_.size() + new2Old.size() );
    leafBoxes_.reserve( leafBoxes_.size() + new2Old.size() );
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
        leaf.verts = mesh.topology.getTriVerts( newFace );
        leafBoxes_.push_back( computeTriBox( mesh.points, leaf.verts ) );
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

void AABBTree::flipOrientation( const FaceBitSet * region )
{
    MR_TIMER;
    ParallelFor( leaves_, [&]( size_t i )
    {
        auto & leaf = leaves_[i];
        if ( !region || region->test( leaf.leafId ) )
            std::swap( leaf.verts[0], leaf.verts[1] );
    } );
}

void AABBTree::updateLeafVerts( const MeshTopology & topology )
{
    MR_TIMER;
    ParallelFor( leaves_, [&]( size_t i )
    {
        leaves_[i].verts = topology.getTriVerts( leaves_[i].leafId );
    } );
}

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
