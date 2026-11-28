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

TEST( MRMesh, PolylineHausdorffDistance )
{
    // two vertical segments at x=0 and x=3
    const Polyline3 a( Contours3f{ { { 0, 0, 0 }, { 0, 0, 1 } }, { { 3, 0, 0 }, { 3, 0, 1 } } } );
    const Polyline3 b( Contour3f{ { 0, 1, 0 }, { 4, 1, 0 } } );

    // the most distant vertex of b is (4, 1, 0) at the distance sqrt(2) from (3, 0, 0)
    EXPECT_FLOAT_EQ( findMaxDistance( a, b ).distSq, 2.0f );
    EXPECT_FLOAT_EQ( findHausdorffDistance( a, b, 0 ).distSq, 2.0f );

    // the most distant point of b is (1.5, 1, 0) inside its segment, equally close to both segments of a
    const float tolerance = 1e-3f;
    for ( auto r : { findHausdorffDistanceOneWay( a, b, tolerance ), findHausdorffDistance( a, b, tolerance ) } )
    {
        EXPECT_LE( r.distSq, 3.25f );
        EXPECT_GE( r.distSq, sqr( std::sqrt( 3.25f ) - tolerance ) );
        const auto pa = a.edgePoint( r.a );
        const auto pb = b.edgePoint( r.b );
        EXPECT_NEAR( pb.x, 1.5f, 0.1f );
        EXPECT_EQ( pb.y, 1.0f );
        EXPECT_NEAR( ( pa - pb ).lengthSq(), r.distSq, 1e-5f );
    }

    // swapped polylines give swapped points
    const auto r = findHausdorffDistance( b, a, tolerance );
    EXPECT_NEAR( b.edgePoint( r.a ).x, 1.5f, 0.1f );
    EXPECT_GE( r.distSq, sqr( std::sqrt( 3.25f ) - tolerance ) );

    // every point of a is as close to b as one of its vertices
    EXPECT_FLOAT_EQ( findHausdorffDistanceOneWay( b, a, tolerance ).distSq, 2.0f );

    // the search stops at the given limit
    const auto lim = findHausdorffDistance( a, b, tolerance, nullptr, 3.0f );
    EXPECT_FLOAT_EQ( lim.distSq, 3.0f );
    EXPECT_FALSE( lim.a.valid() );
    EXPECT_TRUE( lim.b.valid() );

    const Polyline3 empty;
    EXPECT_EQ( findHausdorffDistance( empty, empty, tolerance ).distSq, 0.0f );
    EXPECT_EQ( findHausdorffDistanceOneWay( a, empty, tolerance ).distSq, 0.0f );
    const auto e = findHausdorffDistanceOneWay( empty, b, tolerance, nullptr, 9.0f );
    EXPECT_EQ( e.distSq, 9.0f );
    EXPECT_FALSE( e.a.valid() );
    EXPECT_TRUE( e.b.valid() );
}

} //namespace MR
