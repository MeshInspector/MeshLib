#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include "MRMesh/MRCube.h"
#include "MRMesh/MRConstants.h"
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, MeshDistance)
{
    Mesh sphere1 = makeUVSphere( 1, 8, 8 );

    auto d11 = findDistance( sphere1, sphere1, nullptr, FLT_MAX );
    EXPECT_EQ( d11.distSq, 0 );

    auto zShift = AffineXf3f::translation( Vector3f( 0, 0, 3 ) );
    auto d1z = findDistance( sphere1, sphere1, &zShift, FLT_MAX );
    EXPECT_EQ( d1z.distSq, 1 );

    Mesh sphere2 = makeUVSphere( 2, 8, 8 );

    auto d12 = findDistance( sphere1, sphere2, nullptr, FLT_MAX );
    float dist12 = std::sqrt( d12.distSq );
    EXPECT_TRUE( dist12 > 0.9f && dist12 < 1.0f );
}

TEST( MRMesh, MeshMeshClosestPoint )
{
    auto cube1 = makeCube();
    auto cube2 = makeCube();
    AffineXf3f b2a = AffineXf3f::translation( Vector3f( 0.8f, 0.8f, 0.0f ) ) *
        AffineXf3f::linear( Matrix3f::rotationFromEuler( Vector3f( PI_F * 0.25f, 0.0f, -PI_F * 0.25f ) ) );

    auto sd = findSignedDistance( cube1, cube2, &b2a );
    EXPECT_EQ( sd.signedDist, 0.0f );

    auto usd = findDistance( cube1, cube2, &b2a );
    EXPECT_EQ( usd.distSq, 0.0f );
    EXPECT_TRUE( findProjection( usd.a.proj.point, cube1 ).distSq < 1e-6f );
    EXPECT_TRUE( findProjection( usd.b.proj.point, cube2 ).distSq < 1e-6f );
}

TEST( MRMesh, MaxSurfaceDistance )
{
    // all vertices of the triangle are on the cube, and the center of the cube inside the triangle is the farthest from it
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    const Mesh tri = Mesh::fromTriangles( { Vector3f( 0, 0, 0.5f ), Vector3f( 1, 0, 0.5f ), Vector3f( 0.3f, 1, 0.5f ) }, t );
    const float tol = 1e-3f;

    EXPECT_EQ( findMaxDistanceSqOneWay( cube, tri ), 0 );
    const auto d = std::sqrt( findMaxSurfaceDistanceSqOneWay( cube, tri, tol ) );
    EXPECT_LE( d, 0.5f );
    EXPECT_GE( d, 0.5f - tol );

    // the farthest points of the cube from the triangle are its two vertices with x=1, y=1
    const auto cubeMaxSq = 0.25f + 0.49f / 1.49f;
    EXPECT_NEAR( findMaxSurfaceDistanceSqOneWay( tri, cube, tol ), cubeMaxSq, 1e-6f );
    EXPECT_NEAR( findMaxSurfaceDistanceSq( cube, tri, tol ), cubeMaxSq, 1e-6f );

    // the triangle moved away, and the transformation returning it back
    const auto xf = AffineXf3f::translation( Vector3f( 3, 2, 1 ) ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1, 2, 3 ), 1.0f ) );
    const auto xfInv = xf.inverse();
    Mesh movedTri = tri;
    movedTri.transform( xfInv );
    const auto dXf = std::sqrt( findMaxSurfaceDistanceSqOneWay( cube, movedTri, tol, &xf ) );
    EXPECT_LE( dXf, 0.5f + 1e-5f );
    EXPECT_GE( dXf, 0.5f - tol );
    EXPECT_NEAR( findMaxSurfaceDistanceSq( cube, movedTri, tol, &xf ), cubeMaxSq, 1e-5f );
    EXPECT_NEAR( findMaxSurfaceDistanceSq( movedTri, cube, tol, &xfInv ), cubeMaxSq, 1e-5f );

    EXPECT_EQ( findMaxSurfaceDistanceSqOneWay( cube, tri, tol, nullptr, 0.01f ), 0.01f );
    EXPECT_EQ( findMaxSurfaceDistanceSq( cube, cube, tol ), 0 );
}

} //namespace MR
