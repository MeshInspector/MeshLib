#include <MRMesh/MRSurfaceDistanceBuilder.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>

#include <cmath>

namespace MR
{

TEST(MRMesh, SurfaceDistance)
{
    float vc = 0;
    EXPECT_FALSE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1, vc ) );

    EXPECT_FALSE( getFieldAtC( Vector3f{ 2, 1, 0 }, Vector3f{ 3, 3, 0 }, 1, vc ) );

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 1, 0.5f, 0 }, 1 / std::sqrt(2.0f), vc ) );
    EXPECT_NEAR( vc, 1.5f / std::sqrt(2.0f), 1e-5f );
    vc = 0;
}

} //namespace MR
