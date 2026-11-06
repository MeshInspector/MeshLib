#include "MRAABBTree.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRMesh.h"
#include "MRMeshTriPoint.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include <algorithm>

namespace MR
{

using BoxedFace = BoxedLeaf<FaceTreeTraits3>;

inline Box3f computeFaceBox( const VertCoords & points, const ThreeVertIds & t )
{
    Box3f box;
    box.include( points[t[0]] );
    box.include( points[t[1]] );
    box.include( points[t[2]] );

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
    triangulation_.resize( mp.mesh.topology.faceSize() );
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
        auto & t = triangulation_[f];
        mp.mesh.topology.getTriVerts( f, t );
        boxedFaces[i].box = computeFaceBox( mp.mesh.points, t );
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedFaces ) );
}

void AABBTree::updateTriangulation( const MeshTopology & topology )
{
    MR_TIMER;
    triangulation_.resize( topology.faceSize() );
    ParallelFor( nodes_, [&]( NodeId nid )
    {
        const auto & node = nodes_[nid];
        if ( node.leaf() )
            topology.getTriVerts( node.leafId(), triangulation_[node.leafId()] );
    } );
}

MeshTriPoint AABBTree::toMeshTriPoint( const MeshTopology & topology, FaceId f, const TriPointf & bary ) const
{
    const auto e = topology.edgeWithLeft( f );
    const auto & t = triangulation_[f];
    if ( topology.org( e ) == t[0] && topology.dest( e ) == t[1] )
        return { e, bary };
    const auto v = topology.getLeftTriVerts( e );
    const float w[3] = { 1 - bary.a - bary.b, bary.a, bary.b };
    auto weight = [&]( VertId x ) { return x == t[0] ? w[0] : x == t[1] ? w[1] : w[2]; };
    return { e, TriPointf( weight( v[1] ), weight( v[2] ) ) };
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
        node.box = computeFaceBox( mesh.points, triangulation_[f] );
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

    triangulation_.resize( mesh.topology.faceSize() );
    for ( auto f : oldFaces )
        mesh.topology.getTriVerts( f, triangulation_[f] );
    for ( const auto & [newFace, oldFace] : new2Old )
        mesh.topology.getTriVerts( newFace, triangulation_[newFace] );

    // the boxes of the new nodes from the leaves, and then of the roots
    auto updateBox = [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            node.box = computeFaceBox( mesh.points, triangulation_[node.leafId()] );
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

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
