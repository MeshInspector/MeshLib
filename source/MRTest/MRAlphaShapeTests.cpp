#include <MRMesh/MRAlphaShape.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
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

    auto data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 3_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 4_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 2 ); // two balls touching all three points from the opposite sides are empty

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

} //namespace MR
