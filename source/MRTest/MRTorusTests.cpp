#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRMeshCollide.h>
#include <gtest/gtest.h>
#include <cmath>

namespace MR
{

TEST( MRMesh, SquareTorus )
{
    const float r = 1.0f, s = 0.2f;
    for ( int n : { 3, 16, 128 } )
    {
        for ( float angle : { 0.0f, 0.3f, PI_F / 4 } )
        {
            const auto mesh = makeSquareTorus( r, s, n, angle );
            EXPECT_EQ( mesh.topology.numValidVerts(), 4 * n );
            EXPECT_EQ( mesh.topology.numValidFaces(), 8 * n );
            EXPECT_TRUE( mesh.topology.isClosed() );
            EXPECT_TRUE( findSelfCollidingTriangles( mesh ).value().empty() );

            // each primary segment maps (radial distance, z, t) linearly, so the volume is exact
            const double expected = n * std::sin( 2 * PI / n ) * r * 4 * s * s;
            EXPECT_NEAR( mesh.volume(), expected, 1e-5 * expected );

            const auto box = mesh.computeBoundingBox();
            const float halfHeight = s * ( std::abs( std::cos( angle ) ) + std::abs( std::sin( angle ) ) );
            EXPECT_NEAR( box.max.z, halfHeight, 1e-6f );
            EXPECT_NEAR( box.min.z, -halfHeight, 1e-6f );
        }
    }
}

} //namespace MR
