#include "MRAABBTree.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRMakeSphereMesh.h"
#include "MRBuffer.h"
#include "MRGTest.h"
#include "MRRegionBoundary.h"

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
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)numFaces ),
        [&]( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            FaceId f;
            if ( packed )
                boxedFaces[i].leafId = f = FaceId( i );
            else
                f = boxedFaces[i].leafId;
            boxedFaces[i].box = computeFaceBox( mp.mesh, f );
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedFaces ) );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet & changedVerts )
{
    MR_TIMER;

    const auto changedFaces = getIncidentFaces( mesh.topology, changedVerts );

    // update leaf nodes
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( !node.leaf() )
            return;
        const auto f = node.leafId();
        if ( !changedFaces.test( f ) )
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

template auto AABBTreeBase<FaceTreeTraits3>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<FaceTreeTraits3>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<FaceTreeTraits3>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<FaceTreeTraits3>::getLeafOrderAndReset( LeafBMap & leafMap );

TEST(MRMesh, AABBTree)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTree tree( sphere );
    EXPECT_EQ( tree.nodes().size(), getNumNodes( sphere.topology.numValidFaces() ) );
    EXPECT_EQ( tree[AABBTree::rootNodeId()].box, sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );

    FaceBitSet fs;
    fs.autoResizeSet( 1_f );
    AABBTree smallerTree( { sphere, &fs } );
    EXPECT_EQ( smallerTree.nodes().size(), 1 );
}

TEST(MRMesh, ProjectionToEmptyMesh)
{
    Vector3f p( 1.f, 2.f, 3.f );
    bool hasProjection = Mesh{}.projectPoint( p ).valid();
    EXPECT_FALSE( hasProjection );
}

} //namespace MR
