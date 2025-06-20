#include <MRMesh/MRAABBTreePoints.h>
#include <MRMesh/MRMeshToPointCloud.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, AABBTreePoints )
{
    PointCloud spherePC = meshToPointCloud( makeUVSphere( 1, 8, 8 ) );
    AABBTreePoints tree( spherePC );
    EXPECT_EQ( tree.nodes().size(), getNumNodesPoints( int( spherePC.validPoints.count() ) ) );

    Box3f box;
    for ( auto v : spherePC.validPoints )
        box.include( spherePC.points[v] );

    EXPECT_EQ( tree[AABBTreePoints::rootNodeId()].box, box );

    EXPECT_TRUE( tree[AABBTreePoints::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTreePoints::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );
}

TEST( MRMesh, AABBTreePointsFromMesh )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTreePoints tree( sphere );
    EXPECT_EQ( tree.nodes().size(), getNumNodesPoints( sphere.topology.numValidVerts() ) );

    Box3f box;
    for ( auto v : sphere.topology.getValidVerts() )
        box.include( sphere.points[v] );

    EXPECT_EQ( tree[AABBTreePoints::rootNodeId()].box, box );

    EXPECT_TRUE( tree[AABBTreePoints::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTreePoints::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );
}

} // namespace MR
