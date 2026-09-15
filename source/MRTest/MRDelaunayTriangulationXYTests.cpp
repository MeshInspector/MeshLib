#include "MRMesh/MRDelaunayTriangulationXY.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMesh.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, DelaunayTriangulationXY )
{
    std::vector<Vector3f> points
    {
        { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, // square
        { 0.5f, 0.5f, 1 }, // its center
        { 3, 3, 0 } // far away point, will be invalid in the cloud
    };

    auto mesh = delaunayTriangulationXY( points );
    ASSERT_TRUE( mesh.has_value() );
    EXPECT_EQ( mesh->topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh->topology.numValidFaces(), 6 );

    PointCloud cloud;
    cloud.points.vec_ = points;
    cloud.validPoints.resize( points.size(), true );
    cloud.validPoints.reset( VertId( 5 ) );
    auto cloudMesh = delaunayTriangulationXY( cloud );
    ASSERT_TRUE( cloudMesh.has_value() );
    EXPECT_EQ( cloudMesh->points, cloud.points );
    EXPECT_EQ( cloudMesh->topology.numValidVerts(), 5 );
    EXPECT_FALSE( cloudMesh->topology.hasVert( VertId( 5 ) ) );
    EXPECT_EQ( cloudMesh->topology.numValidFaces(), 4 );
    EXPECT_EQ( cloudMesh->topology.findNumHoles(), 1 );

    auto cloudCopy = cloud;
    auto movedMesh = delaunayTriangulationXY( std::move( cloudCopy ) );
    ASSERT_TRUE( movedMesh.has_value() );
    EXPECT_EQ( movedMesh->points, cloud.points );
    EXPECT_EQ( movedMesh->topology.numValidFaces(), 4 );

    cloud.validPoints.reset( VertId( 4 ) );
    cloud.validPoints.reset( VertId( 3 ) );
    cloud.validPoints.reset( VertId( 2 ) );
    cloudMesh = delaunayTriangulationXY( cloud );
    ASSERT_TRUE( cloudMesh.has_value() );
    EXPECT_EQ( cloudMesh->topology.numValidVerts(), 2 );
    EXPECT_EQ( cloudMesh->topology.numValidFaces(), 0 );
    EXPECT_EQ( cloudMesh->topology.undirectedEdgeSize(), 1 );

    cloud.validPoints.reset( VertId( 1 ) );
    cloudMesh = delaunayTriangulationXY( cloud );
    ASSERT_TRUE( cloudMesh.has_value() );
    EXPECT_EQ( cloudMesh->topology.numValidVerts(), 0 );
    EXPECT_EQ( cloudMesh->topology.undirectedEdgeSize(), 0 );
}

} //namespace MR
