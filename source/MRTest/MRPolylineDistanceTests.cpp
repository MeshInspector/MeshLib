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

    EXPECT_EQ( findMaxDistance( a, a ).distSq, 0.0f );

    // from the vertex (3, 0.5, 0) of b to the end (1, 0, 0) of a
    auto r = findMaxDistanceOneWay( a, b );
    EXPECT_FLOAT_EQ( r.distSq, 4.25f );
    EXPECT_EQ( a.edgePoint( r.a ), Vector3f( 1, 0, 0 ) );
    EXPECT_EQ( b.edgePoint( r.b ), Vector3f( 3, 0.5f, 0 ) );

    // every vertex of a projects inside a segment of b
    r = findMaxDistanceOneWay( b, a );
    EXPECT_FLOAT_EQ( r.distSq, 0.25f );
    EXPECT_EQ( b.edgePoint( r.a ), Vector3f( 0, 0.5f, 0 ) );
    EXPECT_EQ( a.edgePoint( r.b ), Vector3f( 0, 0, 0 ) );

    r = findMaxDistance( a, b );
    EXPECT_FLOAT_EQ( r.distSq, 4.25f );
    EXPECT_EQ( a.edgePoint( r.a ), Vector3f( 1, 0, 0 ) );
    EXPECT_EQ( b.edgePoint( r.b ), Vector3f( 3, 0.5f, 0 ) );

    // the same points, but a and b are swapped
    r = findMaxDistance( b, a );
    EXPECT_FLOAT_EQ( r.distSq, 4.25f );
    EXPECT_EQ( b.edgePoint( r.a ), Vector3f( 3, 0.5f, 0 ) );
    EXPECT_EQ( a.edgePoint( r.b ), Vector3f( 1, 0, 0 ) );

    const auto b2a = AffineXf3f::translation( { 0, -0.5f, 0 } );
    EXPECT_FLOAT_EQ( findMaxDistanceOneWay( a, b, &b2a ).distSq, 4.0f );
    EXPECT_FLOAT_EQ( findMaxDistanceOneWay( b, a, &b2a ).distSq, 1.0f ); // here a is moved away from b
    EXPECT_FLOAT_EQ( findMaxDistance( a, b, &b2a ).distSq, 4.0f );

    // the search stops at the given limit
    r = findMaxDistance( a, b, nullptr, 1.0f );
    EXPECT_FLOAT_EQ( r.distSq, 1.0f );
    EXPECT_FALSE( r.a.valid() );
    EXPECT_TRUE( r.b.valid() );

    const Polyline3 empty;
    r = findMaxDistance( empty, empty );
    EXPECT_EQ( r.distSq, 0.0f );
    EXPECT_FALSE( r.a.valid() );
    EXPECT_FALSE( r.b.valid() );
    EXPECT_EQ( findMaxDistanceOneWay( a, empty ).distSq, 0.0f );

    r = findMaxDistanceOneWay( empty, a, nullptr, 9.0f );
    EXPECT_EQ( r.distSq, 9.0f );
    EXPECT_FALSE( r.a.valid() );
    EXPECT_EQ( a.edgePoint( r.b ), Vector3f( 0, 0, 0 ) );
}

} //namespace MR
