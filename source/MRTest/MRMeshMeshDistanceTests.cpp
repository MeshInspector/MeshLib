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

TEST( MRMesh, MaxDistancePoints )
{
    // the vertices of the tall box with z=2 are the farthest from the cube, and all vertices of the cube are on the tall box
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    const Mesh box = makeCube( Vector3f( 1, 1, 2 ), Vector3f() );

    auto res = findMaxDistanceOneWay( cube, box );
    EXPECT_EQ( res.distSq, 1 );
    EXPECT_EQ( res.distSq, findMaxDistanceSqOneWay( cube, box ) );
    ASSERT_TRUE( res.a.valid() && res.b.valid() );
    EXPECT_EQ( res.b.proj.point.z, 2 );
    EXPECT_EQ( res.a.proj.point, res.b.proj.point - Vector3f( 0, 0, 1 ) );
    EXPECT_EQ( box.triPoint( res.b.mtp( box.topology ) ), res.b.proj.point );
    EXPECT_EQ( res.a.distSq, 1 );
    EXPECT_EQ( res.b.distSq, 1 );

    EXPECT_EQ( findMaxDistanceOneWay( box, cube ).distSq, 0 );

    // the farthest vertex is in the first mesh
    res = findMaxDistance( box, cube );
    EXPECT_EQ( res.distSq, 1 );
    EXPECT_EQ( res.distSq, findMaxDistanceSq( box, cube ) );
    EXPECT_EQ( res.a.proj.point.z, 2 );
    EXPECT_EQ( res.b.proj.point.z, 1 );

    // the point of B is returned in its own space
    const auto xf = AffineXf3f::translation( Vector3f( 0, 0, 3 ) );
    res = findMaxDistance( cube, box, &xf );
    EXPECT_EQ( res.distSq, 16 );
    EXPECT_EQ( res.a.proj.point.z, 1 );
    EXPECT_EQ( res.b.proj.point.z, 2 );

    res = findMaxDistanceOneWay( cube, box, nullptr, 0.25f );
    EXPECT_EQ( res.distSq, 0.25f );
    EXPECT_FALSE( res.a.valid() );
    EXPECT_TRUE( res.b.valid() );
}

} //namespace MR
