#include <MRMesh/MRConstants.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRQuaternion.h>
#include <MRVoxels/MROffset.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, SharpOffsetSelfIntersections )
{
    auto cone = makeCylinderAdvanced( 0.5f, 0.0f, 0.0f, 2 * PI_F, 1.0f, 8 );
    const auto q = Quaternionf( Vector3f::plusZ(), 10 * PI_F / 180 )
        * Quaternionf( Vector3f::plusY(), 20 * PI_F / 180 )
        * Quaternionf( Vector3f::plusX(), 30 * PI_F / 180 );
    cone.transform( AffineXf3f::linear( Matrix3f( q ) ) );

    SharpOffsetParameters params;
    params.voxelSize = 0.05f;
    params.signDetectionMode = SignDetectionMode::ProjectionNormal;
    for ( float offset : { -0.1f, -0.05f, 0.0f } )
    {
        const auto res = sharpOffsetMesh( cone, offset, params );
        ASSERT_TRUE( res.has_value() );
        const auto pairs = findSelfCollidingTriangles( *res );
        ASSERT_TRUE( pairs.has_value() );
        EXPECT_EQ( pairs->size(), 0 ) << "offset " << offset;
    }
}

} //namespace MR
