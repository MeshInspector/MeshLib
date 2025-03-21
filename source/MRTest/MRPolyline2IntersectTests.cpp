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

} //namespace MR
