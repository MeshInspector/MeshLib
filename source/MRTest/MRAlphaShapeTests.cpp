#include <MRMesh/MRAlphaShape.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, AlphaShape )
{
    PointCloud cloud;
    cloud.points.push_back( { 0.5f, 0.5f, 0.1f } ); //0_v
    cloud.points.push_back( { 0.5f, 0.5f, -.1f } ); //1_v
    cloud.points.push_back( { 0,    0,    0 } );    //2_v
    cloud.points.push_back( { 1,    0,    0 } );    //3_v
    cloud.points.push_back( { 0,    1,    0 } );    //4_v
    cloud.validPoints.autoResizeSet( 2_v, 3, true );

    Triangulation tris;
    std::vector<PreciseVertCoords> neis;
    AlphaShapeStats stats;

    auto data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 3_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 4_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 2 ); // two balls touching all three points from the opposite sides are empty

    // only the triangle 2-3-4 is considered, from point #2 with the two others having larger ids
    EXPECT_EQ( stats.consideredTris, 1 );
    EXPECT_EQ( stats.touchableTris, 1 );
    EXPECT_EQ( stats.inBallTests, 0 ); // no other points in the neighbourhood to test

    cloud.validPoints.set( 1_v );
    cloud.invalidateCaches();
    tris.clear();
    data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 1 ); // 1_v is inside one of the two balls

    cloud.validPoints.set( 0_v );
    cloud.invalidateCaches();
    tris.clear();
    data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 0 ); // 0_v and 1_v are inside the balls on the both sides

    const auto allTris = findAlphaShapeAllTriangles( cloud, 3 );
    EXPECT_EQ( allTris.size(), 6 );
}

// four points of a square are exactly on both balls passing via any three of them,
// so every ball emptiness test here is a tie resolved by simulation-of-simplicity
TEST( MRMesh, AlphaShapeSquare )
{
    PointCloud cloud;
    cloud.points.push_back( { 0, 0, 0 } ); //0_v
    cloud.points.push_back( { 1, 0, 0 } ); //1_v
    cloud.points.push_back( { 1, 1, 0 } ); //2_v
    cloud.points.push_back( { 0, 1, 0 } ); //3_v
    cloud.validPoints.autoResizeSet( 0_v, 4, true );

    const auto tris = findAlphaShapeAllTriangles( cloud, 0.8f );
    // the square is covered by two triangles from each side, and the sides take different diagonals;
    // in floating point all the four balls looked empty giving eight triangles
    const std::vector<ThreeVertIds> expectedTris{
        { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v }, // the diagonal 0-2 from the positive side
        { 0_v, 3_v, 1_v }, { 1_v, 3_v, 2_v }  // the diagonal 1-3 from the negative side
    };
    EXPECT_EQ( tris.vec_, expectedTris );

    const auto mesh = findAlphaShape( cloud, 0.8f );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_TRUE( mesh.topology.isClosed() );
}

// two grids crossing along a line: many junction fans where several continuation triangles exist;
// the ccwAroundLine-based selection of the best continuation gives one connected component here,
// while taking the first found continuation gave 468 vertices in 65 components
TEST( MRMesh, AlphaShapeCrossingGrids )
{
    PointCloud cloud;
    for ( int i = 0; i <= 10; ++i )
        for ( int j = 0; j <= 10; ++j )
            cloud.points.push_back( { i * 0.05f, j * 0.05f, 0 } );
    for ( int i = 0; i <= 10; ++i )
        for ( int k = 0; k <= 10; ++k )
            if ( k != 5 )
                cloud.points.push_back( { i * 0.05f, 0.25f, k * 0.05f - 0.25f } );
    cloud.validPoints.autoResizeSet( 0_v, (int)cloud.points.size(), true );

    AlphaShapeStats stats;
    const auto mesh = findAlphaShape( cloud, 0.1f, &stats );
    EXPECT_EQ( stats.consideredTris, 98961 );
    EXPECT_EQ( stats.touchableTris, 47662 );
    // the number of tests depends on the order of the neighbours, in which the first point
    // inside a ball is met, so it is not compared with an exact value here
    EXPECT_GT( stats.inBallTests, stats.touchableTris );
    EXPECT_EQ( mesh.topology.numValidFaces(), 584 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 322 );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
}

} //namespace MR
