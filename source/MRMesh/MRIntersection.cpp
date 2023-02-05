#include "MRIntersection.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, IntersectPlanePlane )
{
    Plane3d plane1( Vector3d( 1, 0, 0 ), 1 );
    Plane3d plane2( Vector3d( 0, 1, 0 ), 1 );
    Plane3d plane3( Vector3d( 0, 0, 1 ), 0 );
    Plane3d plane4( Vector3d( -1, 0, 0 ), 1 );


    auto is0 = intersection( plane1, plane2 );
    ASSERT_TRUE( is0.has_value() );
    ASSERT_NEAR( ( is0->d - Vector3d{ 0., 0., 1. } ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( plane3.project( is0->p ) - Vector3d{ 1., 1., 0. } ).length(), 0., 1e-15 );

    auto is1 = intersection( plane1, plane4 );
    ASSERT_FALSE( is1.has_value() );


    auto dist0 = distance( plane1, plane2 );
    ASSERT_FALSE( dist0.has_value() );

    auto dist1 = distance( plane1, plane4 );
    ASSERT_TRUE( dist1.has_value() );
    ASSERT_NEAR( *dist1, 2., 1e-15 );

}

TEST(MRMesh, IntersectPlaneLine) 
{
    Plane3d plane( Vector3d( 1, 0, 0 ), 1 );
    Line3d line( Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 1 ).normalized() );
    Line3d line1( Vector3d( 0, 0, 0 ), Vector3d( 0, 1, 0 ) );

    auto is0 = intersection( plane, line );
    ASSERT_TRUE( is0.has_value() );
    ASSERT_NEAR( ( *is0 - Vector3d( 1, 1, 1 ) ).length(), 0., 1e-15 );

    auto is1 = intersection( plane, line1 );
    ASSERT_FALSE( is1.has_value() );


    auto dist0 = distance( plane, line );
    ASSERT_FALSE( dist0.has_value() );

    auto dist1 = distance( plane, line1 );
    ASSERT_TRUE( dist1.has_value() );
    ASSERT_NEAR( *dist1, 1., 1e-15 );
}

TEST( MRMesh, IntersectLineLine )
{
    Line3d line1( Vector3d( 1, 0, 0 ), Vector3d( 0, 1, 0 ) );
    Line3d line2( Vector3d( 0, 1, 0 ), Vector3d( 1, 0, 0 ) );
    Line3d line3( Vector3d( 0, 0, 0 ), Vector3d( 0, 0, 1 ) );
    Line3d line4( Vector3d( 0, 1, 0 ), Vector3d( 0, -1, 0 ) );


    auto is0 = intersection( line1, line2 );
    ASSERT_TRUE( is0.has_value() );
    ASSERT_NEAR( ( *is0 - Vector3d{ 1., 1., 0. } ).length(), 0., 1e-15 );

    auto is1 = intersection( line1, line3 );
    ASSERT_FALSE( is1.has_value() );

    auto is2 = intersection( line1, line4 );
    ASSERT_FALSE( is2.has_value() );


    auto dist0 = distance( line1, line2 );
    ASSERT_NEAR( dist0, 0, 1e-15 );

    auto dist1 = distance( line1, line3 );
    ASSERT_NEAR( dist1, 1., 1e-15 );

    auto dist2 = distance( line1, line4 );
    ASSERT_NEAR( dist2, 1., 1e-15 );

    const Line3d line5( Vector3d( 0, 0, 1 ), Vector3d( 1, 1, 0 ).normalized() );
    auto dist15 = distance( line1, line5 );
    ASSERT_NEAR( dist15, 1, 1e-15 );

    auto cl0 = closestPoints( line1, line2 );
    ASSERT_FALSE( cl0.has_value() );

    auto cl1 = closestPoints( line1, line3 );
    ASSERT_TRUE( cl1.has_value() );
    ASSERT_NEAR( ( cl1->a - Vector3d( 1, 0, 0 ) ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( cl1->b - Vector3d( 0, 0, 0 ) ).length(), 0., 1e-15 );

    auto cl2 = closestPoints( line1, line4 );
    ASSERT_FALSE( cl2.has_value() );
}

} //namespace MR
