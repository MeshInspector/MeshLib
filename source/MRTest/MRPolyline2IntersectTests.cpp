#include <MRMesh/MRPolyline2Intersect.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRLine.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRBitSet.h>
#include <random>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, Polyline2RayIntersect )
{
    Vector2f as[2] = { { 0, 1 }, { 4, 5 } };
    Polyline2 polyline;
    polyline.addFromPoints( as, 2, false );

    Line2f line( { 0, 2 }, { 2, -2 } );

    auto res = rayPolylineIntersect( polyline, line );
    ASSERT_TRUE( !!res );
    ASSERT_EQ( res->edgePoint.e, 0_e );
    ASSERT_EQ( res->edgePoint.a, 1.0f / 8 );
    ASSERT_EQ( res->distanceAlongLine, 1.0f / 4 );
}

TEST( MRMesh, Polyline2RayIntersectFloat )
{
    const Line2f lineA( { 21.226973f, -29.397297f }, { 0.9549145f, 0.29688108f} );
    const Line2f lineB( lineA.p + lineA.d.normalized() * 7.f, lineA.d );
    const Contour2f cnt = { { 25.131077f, -16.692158f }, { 28.388832f, -27.170689f }, { 28.726366f, -29.492802f } };
    Polyline2 pl;
    pl.addFromPoints( cnt.data(), cnt.size() );

    const auto projResA = rayPolylineIntersect( pl, lineA );
    EXPECT_TRUE( projResA.has_value() );
    EXPECT_NEAR( projResA->distanceAlongLine, 7.5f, 1e-5f );

    const auto projResB = rayPolylineIntersect( pl, lineB );
    EXPECT_TRUE( projResB.has_value() );
    EXPECT_NEAR( projResB->distanceAlongLine, 0.5f, 1e-5f );
}

TEST( MRMesh, IsPointInsidePolyline )
{
    Vector2f as[9] = {
        { -1, -1 },
        { -1, 0 },
        { -1, 1 },
        { 0, 1 },
        { 1, 1 },
        { 1, 0 },
        { 1, -1 },
        { 0, -1 },
        { -1, -1 }
    };
    Polyline2 polyline;
    polyline.addFromPoints( as, 9, true );

    // it is expected to have only one intersection with 
    ASSERT_TRUE( isPointInsidePolyline( polyline, Vector2f( 0, 0 ) ) );
}

TEST( MRMesh, FindGridPointsInsidePolyline )
{
    std::mt19937 rnd( 1 );
    for ( int n = 0; n < 40; ++n )
    {
        // random self-intersecting polygons, half of them with integer vertices to get rows and grid points exactly on vertices and edges
        const bool intVerts = n % 2 == 0;
        std::uniform_real_distribution<float> coord( -10, 40 );
        Contour2f cont( 3 + rnd() % 30 );
        for ( auto & p : cont )
        {
            p = { coord( rnd ), coord( rnd ) };
            if ( intVerts )
                p = { std::round( p.x ), std::round( p.y ) };
        }
        cont.push_back( cont.front() );
        const Polyline2 polyline( { cont } );

        const Vector2i dims( 37, 29 );
        const Vector2f origin = n % 4 < 2 ? Vector2f() : Vector2f( -3.5f, 1.25f );
        const Vector2f step = n % 8 < 4 ? Vector2f::diagonal( 1 ) : Vector2f( 0.75f, 1.5f );
        const auto bits = findGridPointsInsidePolyline( polyline, dims, origin, step );
        ASSERT_EQ( bits.size(), size_t( dims.x ) * dims.y );
        for ( int y = 0; y < dims.y; ++y )
            for ( int x = 0; x < dims.x; ++x )
                ASSERT_EQ( bits.test( x + y * dims.x ), isPointInsidePolyline( polyline, { step.x * float( x ) + origin.x, step.y * float( y ) + origin.y } ) )
                    << "polygon " << n << ", point " << x << ", " << y;
    }
}

} //namespace MR
