#include <MRMesh/MRBestFit.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRLine3.h>
#include <MRMesh/MRPlane3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, FindBestLineSimple )
{
    PointAccumulator acc;
    acc.addPoint( Vector3d{ 0, 0, 0 } );
    acc.addPoint( Vector3d{ 1, 0, 0 } );
    acc.addPoint( Vector3d{ 2, 0, 0 } );
    acc.addPoint( Vector3d{ 3, 0, 0 } );
    const Line3d line = acc.getBestLine().normalized();

    const double deltaD = std::abs( dot( line.d, Vector3d( 1, 0, 0 ) ) ) - 1.;
    ASSERT_LE( deltaD, 1e-12 );
    // [ x - p, d ] = 0   =>   [ line.p - 0, d ] = 0
    const double deltaP = cross( line.p, line.d ).length();
    ASSERT_LE( deltaP, 1e-12 );
}

TEST( MRMesh, FindBestLineFull )
{
    PointAccumulator acc;
    const Vector3d e1 = Vector3d( 1, 1, 1 ).normalized();
    const Vector3d e2 = Vector3d( 1, -1, 0 ).normalized();
    const Vector3d e3 = Vector3d( -1, -1, 2 ).normalized();

    const Vector3d de1{ e1 * 100. };
    const Vector3d de2{ e2 * 2. };
    const Vector3d de3{ e3 * 2.5 };
    const Vector3d shift{ e1 * 7. + e2 * 11. - e3 * 9.3 };

    for ( int i = 0; i < 8; ++i )
    {
        acc.addPoint( de1 * ( ( i & 1 ) / 1 * 2. - 1. ) +
                      de2 * ( ( i & 2 ) / 2 * 2. - 1. ) +
                      de3 * ( ( i & 4 ) / 4 * 2. - 1. ) + shift );
    }

    const Line3d line = acc.getBestLine().normalized();

    const double deltaD = std::abs( dot( line.d, e1 ) ) - 1.;
    ASSERT_LE( deltaD, 1e-12 );
    // [ x - p, d ] = 0   =>   [ line.p - shift, line.d ] = 0
    const double deltaP = cross( line.p - shift, line.d ).length();
    ASSERT_LE( deltaP, 1e-12 );
}

TEST( MRMesh, FindBestPlaneSimple )
{
    PointAccumulator acc;
    acc.addPoint( Vector3d{ 0, 0, 0 } );
    acc.addPoint( Vector3d{ 0, 1, 0 } );
    acc.addPoint( Vector3d{ 1, 1, 0 } );
    acc.addPoint( Vector3d{ 1, 0, 0 } );
    auto plane = acc.getBestPlane();

    ASSERT_EQ( plane.n, ( Vector3d{ 0, 0, 1 } ) );
    ASSERT_EQ( plane.d, 0 );
}

TEST( MRMesh, FindBestPlaneFull )
{
    PointAccumulator acc;
    const Vector3d e1 = Vector3d( 1, 1, 1 ).normalized();
    const Vector3d e2 = Vector3d( 1, -1, 0 ).normalized();
    const Vector3d e3 = Vector3d( -1, -1, 2 ).normalized();

    const Vector3d de1{ e1 * 7.5 };
    const Vector3d de2{ e2 * 31. };
    const Vector3d de3{ e3 * 43.3 };
    const Vector3d shift{ e1 * 7. + e2 * 11. - e3 * 9.3 };

    for ( int i = 0; i < 8; ++i )
    {
        acc.addPoint( de1 * ( ( i & 1 ) / 1 * 2. - 1. ) +
                      de2 * ( ( i & 2 ) / 2 * 2. - 1. ) +
                      de3 * ( ( i & 4 ) / 4 * 2. - 1. ) + shift );
    }

    const Plane3d plane = acc.getBestPlane().normalized();

    const double deltaN = std::abs( dot( plane.n, e1 ) ) - 1.;
    ASSERT_LE( deltaN, 1e-12 );
    const double deltaD = plane.d - dot( shift, plane.n );
    ASSERT_LE( deltaD, 1e-12 );
}

TEST( MRMesh, FindBestPlaneRealData )
{
    PointAccumulator acc;
    std::vector<Vector3d> pnts =
    {
        { -0.00731878, 0.0188899, -0.0299074 },
        { 0.01220270, 0.0189450, -0.0270888 },
        { -0.01609020, 0.0179431, -0.0238381 },
        { 0.01530080, 0.0187438, -0.0128305 },
        { -0.01822650, 0.0168881, -0.0062546 },
        { 0.00747610, 0.0177175, -0.0304660 },
        { 0.01389150, 0.0179392, -0.0290745 },
        { 0.01850000, 0.0183350, -0.0255000 },
        { 0.01886230, 0.0179144, -0.0171638 },
        { 0.01981160, 0.0169063, -0.0101996 },
        { 0.02181320, 0.0159555, -0.0019617 },
        { 0.02597120, 0.0158570,  0.0188537 }
    };
    for ( const auto& p : pnts )
        acc.addPoint( p );

    auto bestPlane = acc.getBestPlane();
    auto anotherPlane = Plane3d::fromDirAndPt( { 0.0352121070, 0.999376059, 0.00275902473 }, { -0.0182264727, 0.0168880932, -0.00625459058 } );
    double bestSumSq = 0, anotherSumSq = 0;
    for ( const auto& p : pnts )
    {
        bestSumSq += sqr( bestPlane.distance( p ) );
        anotherSumSq += sqr( anotherPlane.distance( p ) );
    }

    ASSERT_LE( bestSumSq, anotherSumSq );
}

TEST( MRMesh, Torus3 )
{
    const Torus3d t( Vector3d{}, Vector3d( 0, 0, 1 ), 3, 1 );
    EXPECT_NEAR( t.distance( { 5, 0, 0 } ), 1, 1e-15 );
    EXPECT_NEAR( t.distance( { 0, 3, 0.5 } ), -0.5, 1e-15 );
    EXPECT_NEAR( t.distance( { 0, 0, 0 } ), 2, 1e-15 );
    EXPECT_NEAR( ( t.project( { 5, 0, 0 } ) - Vector3d( 4, 0, 0 ) ).length(), 0, 1e-15 );
    EXPECT_NEAR( ( t.normal( { 0, 3, 0.5 } ) - Vector3d( 0, 0, 1 ) ).length(), 0, 1e-15 );
}

TEST( MRMesh, FindBestSphereCap )
{
    // a cap of 0.3 radians
    const Vector3d c( 3, -2, 5 ), dir = Vector3d( 1, 2, 3 ).normalized();
    const auto [u, v] = dir.perpendicular();
    std::vector<Vector3d> pts;
    for ( int i = 0; i < 10; ++i )
        for ( int j = 0; j < 10; ++j )
        {
            const double theta = 0.03 * i, phi = 0.2 * PI * j;
            pts.push_back( c + 7.0 * ( std::cos( theta ) * dir + std::sin( theta ) * ( std::cos( phi ) * u + std::sin( phi ) * v ) ) );
        }
    const auto s = findBestSphere( pts );
    ASSERT_TRUE( s );
    EXPECT_NEAR( ( s->center - c ).length(), 0, 1e-9 );
    EXPECT_NEAR( s->radius, 7, 1e-9 );
}

TEST( MRMesh, FindBestCircleShortArc )
{
    // 5 degrees of a circle far from the origin
    const Vector2d c( 1000, -500 );
    std::vector<Vector2d> pts;
    for ( int i = 0; i < 20; ++i )
    {
        const double a = 0.3 + i * ( 5 * PI / 180 ) / 19;
        pts.push_back( c + 300.0 * Vector2d( std::cos( a ), std::sin( a ) ) );
    }
    const auto s = findBestCircle( pts );
    ASSERT_TRUE( s );
    EXPECT_NEAR( ( s->center - c ).length(), 0, 1e-6 );
    EXPECT_NEAR( s->radius, 300, 1e-6 );

    // no points, no circle
    EXPECT_FALSE( findBestCircle( {} ) );
}

TEST( MRMesh, FindBestCylinderByNormals )
{
    const Line3d axis( Vector3d( 2, 3, -1 ), Vector3d( 1, 1, 0.3 ).normalized() );
    const auto [u, v] = axis.d.perpendicular();
    std::vector<Vector3d> pts, normals;
    std::vector<Line3d> normalLines;
    for ( int i = 0; i < 12; ++i )
        for ( int j = 0; j < 5; ++j )
        {
            const double phi = 0.5 * i, h = -5 + 2.5 * j;
            const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
            pts.push_back( axis.p + h * axis.d + 4.0 * er );
            normals.push_back( er );
            normalLines.emplace_back( pts.back(), er );
        }
    const auto cyl = findBestCylinderByNormals( pts, normals );
    ASSERT_TRUE( cyl );
    EXPECT_NEAR( cross( cyl->direction(), axis.d ).length(), 0, 1e-9 );
    EXPECT_NEAR( cyl->radius, 4, 1e-9 );
    EXPECT_NEAR( cyl->length, 10, 1e-9 );
    EXPECT_NEAR( ( cyl->center() - axis.p ).length(), 0, 1e-9 );

    // the lines of the normals of a plane are crossed by all lines along the normal and by the line at infinity: any of them can be returned
    std::vector<Line3d> planeNormals;
    for ( int i = 0; i < 5; ++i )
        planeNormals.emplace_back( Vector3d( i, i * i, 0 ), Vector3d( 0, 0, 1 ) );
    if ( const auto pa = findRevolutionAxis( planeNormals ) )
    {
        EXPECT_NEAR( cross( pa->d, Vector3d( 0, 0, 1 ) ).length(), 0, 1e-6 );
    }

    const auto ra = findRevolutionAxis( normalLines );
    ASSERT_TRUE( ra );
    EXPECT_NEAR( cross( ra->d, axis.d ).length(), 0, 1e-9 );
    EXPECT_NEAR( axis.distanceSq( ra->p ), 0, 1e-18 );
}

TEST( MRMesh, FindBestCone )
{
    const Vector3d apex( 1, 2, 3 ), a = Vector3d( 0.2, -0.5, 1 ).normalized();
    const double angle = 0.4;
    const auto [u, v] = a.perpendicular();
    // one strip of the cone between the heights 5 and 5.5
    std::vector<Vector3d> pts;
    std::vector<Plane3d> planes;
    for ( int i = 0; i < 30; ++i )
        for ( double h : { 5.0, 5.5 } )
        {
            const double phi = 2 * PI * i / 30;
            const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
            pts.push_back( apex + h * a + h * std::tan( angle ) * er );
            planes.push_back( Plane3d::fromDirAndPt( er * std::cos( angle ) - a * std::sin( angle ), pts.back() ) );
        }
    const auto cone = findBestConeByTangentPlanes( pts, planes );
    ASSERT_TRUE( cone );
    EXPECT_NEAR( ( cone->apex() - apex ).length(), 0, 1e-9 );
    EXPECT_NEAR( ( cone->direction() - a ).length(), 0, 1e-9 );
    EXPECT_NEAR( cone->angle, angle, 1e-9 );
    EXPECT_NEAR( cone->height, 5.5, 1e-9 );

    // given axis with a not-unit direction pointing away from the points
    const auto cone2 = findBestConeWithAxis( pts, Line3d( apex + 3.0 * a, -2.0 * a ) );
    ASSERT_TRUE( cone2 );
    EXPECT_NEAR( ( cone2->apex() - apex ).length(), 0, 1e-9 );
    EXPECT_NEAR( ( cone2->direction() - a ).length(), 0, 1e-9 );
    EXPECT_NEAR( cone2->angle, angle, 1e-9 );
}

TEST( MRMesh, FindBestTorus )
{
    const Torus3d t( Vector3d( -1, 4, 2 ), Vector3d( 0.3, 0.4, 1 ).normalized(), 5, 1.5 );
    const auto [u, v] = t.axis.perpendicular();
    // a patch of the torus like a fillet: a quarter of the revolution and 2 radians of the tube
    std::vector<Vector3d> pts;
    std::vector<Line3d> normalLines;
    for ( int i = 0; i < 10; ++i )
        for ( int j = 0; j < 10; ++j )
        {
            const double phi = 0.15 * i, psi = 0.2 * j;
            const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
            const Vector3d n = std::cos( psi ) * er + std::sin( psi ) * t.axis;
            pts.push_back( t.center + t.majorRadius * er + t.minorRadius * n );
            normalLines.emplace_back( pts.back(), n );
        }
    auto check = [&]( const Torus3d& res, double eps )
    {
        EXPECT_NEAR( ( res.center - t.center ).length(), 0, eps );
        EXPECT_NEAR( cross( res.axis, t.axis ).length(), 0, eps );
        EXPECT_NEAR( res.majorRadius, t.majorRadius, eps );
        EXPECT_NEAR( res.minorRadius, t.minorRadius, eps );
    };

    const auto axis = findRevolutionAxis( normalLines );
    ASSERT_TRUE( axis );
    const auto res = findBestTorusWithAxis( pts, *axis );
    ASSERT_TRUE( res );
    check( *res, 1e-9 );

    const Torus3d initial( t.center + Vector3d( 0.05, -0.03, 0.02 ), ( t.axis + Vector3d( 0.02, 0, -0.01 ) ).normalized(), 5.1, 1.45 );
    check( refineTorus( pts, initial ), 1e-7 );
}

} // namespace MR
