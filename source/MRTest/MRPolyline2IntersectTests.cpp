#include <MRMesh/MRPolyline2Intersect.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRLine.h>
#include <MRMesh/MRGTest.h>

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

TEST( MRMesh, Polyline2RayIntersectDouble )
{
    const Line2d lineA( { 21.226973, -29.397297 }, { 0.9549145, 0.29688108 } );
    const Line2d lineB( lineA.p + lineA.d.normalized() * 7., lineA.d );
    const Contour2f cnt = { { 25.131077f, -16.692158f }, { 28.388832f, -27.170689f }, { 28.726366f, -29.492802f } };
    Polyline2 pl;
    pl.addFromPoints( cnt.data(), cnt.size() );

    const auto projResA = rayPolylineIntersect( pl, lineA ); //no intersection in floats
    EXPECT_TRUE( projResA.has_value() );
    EXPECT_NEAR( projResA->distanceAlongLine, 7.5f, 1e-6f );

    const auto projResB = rayPolylineIntersect( pl, lineB );
    EXPECT_TRUE( projResB.has_value() );
    EXPECT_NEAR( projResB->distanceAlongLine, 0.5f, 1e-6f );
}

} //namespace MR
