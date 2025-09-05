#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include "MRMesh/MRCube.h"
#include "MRMesh/MRConstants.h"
#include <MRMesh/MRGTest.h>

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
    EXPECT_TRUE( findProjection( usd.a.point, cube1 ).distSq < 1e-6f );
    EXPECT_TRUE( findProjection( usd.b.point, cube2 ).distSq < 1e-6f );
}

} //namespace MR
