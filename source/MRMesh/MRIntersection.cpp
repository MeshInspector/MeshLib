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


    auto dist0 = closestPoints( line1, line2 ).length();
    ASSERT_NEAR( dist0, 0, 1e-15 );

    auto dist1 = closestPoints( line1, line3 ).length();
    ASSERT_NEAR( dist1, 1., 1e-15 );

    auto dist2 = closestPoints( line1, line4 ).length();
    ASSERT_NEAR( dist2, 1., 1e-15 );

    const Line3d line5( Vector3d( 0, 0, 1 ), Vector3d( 1, 1, 0 ).normalized() );
    auto dist15 = closestPoints( line1, line5 ).length();
    ASSERT_NEAR( dist15, 1, 1e-15 );

    auto cl0 = closestPoints( line1, line2 );
    ASSERT_NEAR( ( cl0.a - Vector3d( 1, 1, 0 ) ).length(), 0, 1e-15 );
    ASSERT_NEAR( ( cl0.b - Vector3d( 1, 1, 0 ) ).length(), 0, 1e-15 );

    auto cl1 = closestPoints( line1, line3 );
    ASSERT_NEAR( ( cl1.a - Vector3d( 1, 0, 0 ) ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( cl1.b - Vector3d( 0, 0, 0 ) ).length(), 0., 1e-15 );

    auto cl2 = closestPoints( line1, line4 );
    ASSERT_NEAR( ( cl2.a - cl2.b - Vector3d( 1, 0, 0 ) ).length(), 0., 1e-15 );

    auto cl15 = closestPoints( line1, line5 );
    ASSERT_NEAR( ( cl15.a - Vector3d( 1, 1, 0 ) ).length(), 0, 1e-15 );
    ASSERT_NEAR( ( cl15.b - Vector3d( 1, 1, 1 ) ).length(), 0, 1e-15 );
}

TEST( MRMesh, IntersectLineShere )
{
    Line3d line1( Vector3d( 2, 0, 2 ), Vector3d( 0, 1, 0 ) );
    Line3d line2( Vector3d( 4, 0, 2 ), Vector3d( 0, 1, 0 ) );
    Sphere3d sphere( Vector3d( 2, 2, 2 ), 1 );
    auto is1 = intersection( line1, sphere );
    ASSERT_TRUE( is1 );
    ASSERT_NEAR( is1->first, 1., 1e-15 );
    ASSERT_NEAR( is1->second, 3., 1e-15 );

    auto is2 = intersection( line2, sphere );
    ASSERT_FALSE( is2 );
}

TEST( MRMesh, ClosestPointsLine3Box3 )
{
    {
        auto cp = closestPoints( Line3f{ Vector3f{}, Vector3f{1,0,0} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{1,0,0} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{1,1,1} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{}, Vector3f{0,1,0} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{0,1,0} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{1,1,1} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{}, Vector3f{0,0,1} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{0,0,1} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{1,1,1} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{}, Vector3f{-1,1,0} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{0,0,0} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{1,1,1} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{3,5,4}, Vector3f{-1,1,0} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{4,4,4} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{3,3,3} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{4,0,0}, Vector3f{0,1,-1} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{4,0,0} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{3,1,1} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{0,5,3}, Vector3f{0,1,-1} }, Box3f{ Vector3f{1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{0,4,4} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{1,3,3} ).length() < 1e-6f );
    }
    {
        auto cp = closestPoints( Line3f{ Vector3f{0,2,-2}, Vector3f{0,1,-1} }, Box3f{ Vector3f{-1,1,1}, Vector3f{3,3,3} } );
        ASSERT_TRUE( ( cp.a - Vector3f{0,0,0} ).length() < 1e-6f );
        ASSERT_TRUE( ( cp.b - Vector3f{0,1,1} ).length() < 1e-6f );
    }
}

} //namespace MR
