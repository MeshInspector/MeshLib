#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include "MRMesh/MRCube.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include <MRMesh/MRGTest.h>

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
    EXPECT_NEAR( dist.signedDist, std::sqrt( upDistLimitSq ), 1e-6f );
}

} //namespace MR
