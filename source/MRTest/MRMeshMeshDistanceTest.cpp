#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include "MRMesh/MRCube.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, findSignedDistanceTest )
{
    auto a = makeCube( Vector3f::diagonal( 2.0f ), Vector3f::diagonal( -1.0f ) );
    auto b = makeCube( Vector3f::diagonal( 1.0f ), Vector3f::diagonal( -0.5f ) );

    auto dist = findSignedDistance( a, b );
    EXPECT_TRUE( dist.signedDist < 0.0f );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::BInside );

    AffineXf3f xf;
    xf.b.z = 0.6f;
    dist = findSignedDistance( a, b, &xf );
    EXPECT_TRUE( dist.signedDist < 0.0f );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::Colliding );

    xf.b.z = 1.4f;
    dist = findSignedDistance( a, b, &xf );
    EXPECT_TRUE( dist.signedDist < 0.0f );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::Colliding );

    xf.b.z = 1.5f;
    dist = findSignedDistance( a, b, &xf );
    EXPECT_TRUE( dist.signedDist == 0.0f );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::Touching );


    xf.b.z = 1.6f;
    dist = findSignedDistance( a, b, &xf );
    EXPECT_TRUE( dist.signedDist > 0.0f );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::BothOutside );

    // upDistLimitSq smaller than the real distSq: must short-circuit to BothOutside
    // with signedDist == sqrt(upDistLimitSq), not run collision checks on the sentinel result
    const float upDistLimitSq = 0.001f; // xf.b.z = 1.6f gives real distSq = 0.01
    dist = findSignedDistance( a, b, &xf, upDistLimitSq );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::NotColliding );
    EXPECT_FALSE( dist.a );
    EXPECT_FALSE( dist.b );
    EXPECT_NEAR( dist.signedDist, std::sqrt( upDistLimitSq ), 1e-6f );
}

TEST( MRMesh, findSignedDistanceValidPointsOnCollision )
{
    // Regression: when two meshes touch at several vertices (dist == 0) without
    // any vert strictly inside the other, the AB loop in findSignedDistance picks
    // up a default invalid PointOnFace from a ring-expanded vertBS entry (its
    // dist == 0 beats the FLT_MAX init), and the BA loop's stored dist == 0
    // entries cannot improve on that minimum. Result: signedRes.b stays invalid.
    // The fix guards the assignment with proj.valid(), so the BA loop's valid
    // dist == 0 entry wins instead.
    auto a = makeCube( Vector3f::diagonal( 2.0f ), Vector3f::diagonal( -1.0f ) );
    auto b = makeCube( Vector3f::diagonal( 2.0f ), Vector3f::diagonal( -1.0f ) );

    AffineXf3f xf;
    xf.A = Matrix3f::rotation( Vector3f( 1.0f, 0.0f, 0.0f ), 0.05f ); // small x-axis tilt
    xf.b = Vector3f( 0.0f, 0.0f, 2.0f );                              // b above a, touching at z=1 line
    auto dist = findSignedDistance( a, b, &xf );
    EXPECT_TRUE( dist.status == MeshMeshCollisionStatus::Colliding );
    EXPECT_TRUE( dist.a );
    EXPECT_TRUE( dist.b );
}

} //namespace MR
