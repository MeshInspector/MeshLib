#include <MRMesh/MRPolylineDistance.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, PolylineMaxDistance )
{
    const Polyline3 a( Contour3f{ { 0, 0, 0 }, { 1, 0, 0 } } );
    const Polyline3 b( Contour3f{ { 0, 0.5f, 0 }, { 0.5f, 0.5f, 0 }, { 3, 0.5f, 0 } } );

    EXPECT_EQ( findMaxDistanceSq( a, a ), 0.0f );

    // from the vertex (3, 0.5, 0) of b to the end (1, 0, 0) of a
    EXPECT_FLOAT_EQ( findMaxDistanceSqOneWay( a, b ), 4.25f );
    // every vertex of a projects inside a segment of b
    EXPECT_FLOAT_EQ( findMaxDistanceSqOneWay( b, a ), 0.25f );
    EXPECT_FLOAT_EQ( findMaxDistanceSq( a, b ), 4.25f );
    EXPECT_FLOAT_EQ( findMaxDistanceSq( b, a ), 4.25f );

    const auto b2a = AffineXf3f::translation( { 0, -0.5f, 0 } );
    EXPECT_FLOAT_EQ( findMaxDistanceSqOneWay( a, b, &b2a ), 4.0f );
    EXPECT_FLOAT_EQ( findMaxDistanceSqOneWay( b, a, &b2a ), 1.0f ); // here a is moved away from b
    EXPECT_FLOAT_EQ( findMaxDistanceSq( a, b, &b2a ), 4.0f );

    // the search stops at the given limit
    EXPECT_FLOAT_EQ( findMaxDistanceSq( a, b, nullptr, 1.0f ), 1.0f );

    const Polyline3 empty;
    EXPECT_EQ( findMaxDistanceSq( empty, empty ), 0.0f );
    EXPECT_EQ( findMaxDistanceSqOneWay( a, empty ), 0.0f );
    EXPECT_EQ( findMaxDistanceSqOneWay( empty, a, nullptr, 9.0f ), 9.0f );
}

} //namespace MR
