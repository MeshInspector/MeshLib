#include <MRMesh/MRBestFit.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRLine3.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRVector.h>
#include <gtest/gtest.h>
#include <random>

namespace MR
{

namespace
{

/// uniform noise in [-amp, amp) from the raw output of the generator, the same on all platforms
double noise( std::mt19937& gen, double amp )
{
    return amp * ( double( gen() ) / 2147483648.0 - 1 );
}

} // anonymous namespace

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
    EXPECT_EQ( Torus3d( Torus3f( t ) ), t );
}

TEST( MRMesh, FindBestSphereCap )
{
    // a cap of 0.3 radians
    const Vector3d c( 3, -2, 5 ), dir = Vector3d( 1, 2, 3 ).normalized();
    const auto [u, v] = dir.perpendicular();
    Vector<Vector3d> pts, normals;
    Vector<Line3d> normalLines;
    for ( int i = 0; i < 10; ++i )
        for ( int j = 0; j < 10; ++j )
        {
            const double theta = 0.03 * i, phi = 0.2 * PI * j;
            normals.push_back( std::cos( theta ) * dir + std::sin( theta ) * ( std::cos( phi ) * u + std::sin( phi ) * v ) );
            pts.push_back( c + 7.0 * normals.back() );
            normalLines.emplace_back( pts.back(), normals.back() );
        }
    const auto s = findBestSphere( pts );
    ASSERT_TRUE( s );
    EXPECT_NEAR( ( s->center - c ).length(), 0, 1e-9 );
    EXPECT_NEAR( s->radius, 7, 1e-9 );

    // all lines through the center cross the normals, and no direction is orthogonal to them
    EXPECT_FALSE( findRevolutionAxis( normalLines ) );
    EXPECT_FALSE( findBestCylinderByNormals( pts, normals ) );

    // the points of a circle are in one plane: every sphere through the circle passes through them, and the minimum-norm one is centered at it
    Vector<Vector3d> circle;
    for ( int i = 0; i < 4; ++i )
        circle.push_back( c + 7.0 * ( std::cos( 1.5 * i ) * u + std::sin( 1.5 * i ) * v ) );
    const auto cs = findBestSphere( circle );
    ASSERT_TRUE( cs );
    EXPECT_NEAR( ( cs->center - c ).length(), 0, 1e-9 );
    EXPECT_NEAR( cs->radius, 7, 1e-9 );

    // no torus fits the points of a sphere better than the sphere itself: the refined torus degenerates into it
    const auto rt = refineTorus( pts, Torus3d( c + Vector3d( 0.1, 0.05, -0.1 ), dir, 0.5, 6.4 ) );
    EXPECT_LT( rt.majorRadius, 1e-3 );
    EXPECT_NEAR( rt.minorRadius, 7, 1e-3 );
    EXPECT_NEAR( ( rt.center - c ).length(), 0, 1e-3 );
}

TEST( MRMesh, FindBestCircleShortArc )
{
    // 5 degrees of a circle far from the origin
    const Vector2d c( 1000, -500 );
    Vector<Vector2d> pts;
    for ( int i = 0; i < 20; ++i )
    {
        const double a = 0.3 + i * ( 5 * PI / 180 ) / 19;
        pts.push_back( c + 300.0 * Vector2d( std::cos( a ), std::sin( a ) ) );
    }
    const auto s = findBestCircle( pts );
    ASSERT_TRUE( s );
    EXPECT_NEAR( ( s->center - c ).length(), 0, 1e-6 );
    EXPECT_NEAR( s->radius, 300, 1e-6 );

    // no points or coincident points, no circle; symmetric collinear points give the circle centered at their mean
    EXPECT_FALSE( findBestCircle( {} ) );
    EXPECT_FALSE( findBestCircle( { { 1, 2 }, { 1, 2 } } ) );
    const auto line = findBestCircle( { { 1, 2 }, { 3, 3 }, { 5, 4 } } );
    ASSERT_TRUE( line );
    EXPECT_NEAR( ( line->center - Vector2d( 3, 3 ) ).length(), 0, 1e-12 );
    EXPECT_NEAR( line->radius, std::sqrt( 10.0 / 3 ), 1e-12 );
}

TEST( MRMesh, FindBestCylinderByNormals )
{
    const Line3d axis( Vector3d( 2, 3, -1 ), Vector3d( 1, 1, 0.3 ).normalized() );
    const auto [u, v] = axis.d.perpendicular();
    Vector<Vector3d> pts, normals;
    Vector<Line3d> normalLines;
    Vector<Plane3d> planes;
    for ( int i = 0; i < 12; ++i )
        for ( int j = 0; j < 5; ++j )
        {
            const double phi = 0.5 * i, h = -5 + 2.5 * j;
            const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
            pts.push_back( axis.p + h * axis.d + 4.0 * er );
            normals.push_back( er );
            normalLines.emplace_back( pts.back(), er );
            planes.push_back( Plane3d::fromDirAndPt( er, pts.back() ) );
        }
    const auto cyl = findBestCylinderByNormals( pts, normals );
    ASSERT_TRUE( cyl );
    EXPECT_NEAR( cross( cyl->direction(), axis.d ).length(), 0, 1e-9 );
    EXPECT_NEAR( cyl->radius, 4, 1e-9 );
    EXPECT_NEAR( cyl->length, 10, 1e-9 );
    EXPECT_NEAR( ( cyl->center() - axis.p ).length(), 0, 1e-9 );

    const auto ra = findRevolutionAxis( normalLines );
    ASSERT_TRUE( ra );
    EXPECT_NEAR( cross( ra->d, axis.d ).length(), 0, 1e-9 );
    EXPECT_NEAR( axis.distanceSq( ra->p ), 0, 1e-18 );

    // the tangent planes of a cylinder define no cone, also on a strip of half a degree with nearly parallel normals
    EXPECT_FALSE( findBestConeByTangentPlanes( pts, planes ) );
    planes.clear();
    for ( int i = 0; i < 6; ++i )
    {
        const Vector3d er = std::cos( 0.7 + 0.0017 * i ) * u + std::sin( 0.7 + 0.0017 * i ) * v;
        planes.push_back( Plane3d::fromDirAndPt( er, axis.p + 20.0 * er ) );
    }
    EXPECT_FALSE( findBestConeByTangentPlanes( pts, planes ) );

    // all lines along the normal of a plane cross its normals, and all directions in the plane are orthogonal to them
    Vector<Vector3d> planePts, planeNormals;
    Vector<Line3d> planeNormalLines;
    for ( int i = 0; i < 5; ++i )
    {
        planePts.emplace_back( i, i * i, 0 );
        planeNormals.emplace_back( 0, 0, 1 );
        planeNormalLines.emplace_back( planePts.back(), planeNormals.back() );
    }
    EXPECT_FALSE( findRevolutionAxis( planeNormalLines ) );
    EXPECT_FALSE( findBestCylinderByNormals( planePts, planeNormals ) );
}

TEST( MRMesh, FindRevolutionAxisNoisy )
{
    // a quarter of a cylinder of radius 50 and length 20 with noisy normals, many times
    const Line3d axis( Vector3d( 2, 3, -1 ), Vector3d( 1, 1, 0.3 ).normalized() );
    const auto [u, v] = axis.d.perpendicular();
    std::mt19937 gen( 0 );
    double maxAngle = 0;
    for ( int k = 0; k < 50; ++k )
    {
        Vector<Line3d> normalLines;
        for ( int i = 0; i < 12; ++i )
            for ( int j = 0; j < 5; ++j )
            {
                const double phi = 0.5 * PI * i / 11, h = -10 + 5.0 * j;
                const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
                const double nx = noise( gen, 0.01 );
                const double ny = noise( gen, 0.01 );
                const double nz = noise( gen, 0.01 );
                normalLines.emplace_back( axis.p + h * axis.d + 50.0 * er, ( er + Vector3d( nx, ny, nz ) ).normalized() );
            }
        const auto ra = findRevolutionAxis( normalLines );
        ASSERT_TRUE( ra );
        maxAngle = std::max( maxAngle, cross( ra->d, axis.d ).length() );
    }
    EXPECT_LT( maxAngle, 0.05 );
}

TEST( MRMesh, FindBestCone )
{
    const Vector3d apex( 1, 2, 3 ), a = Vector3d( 0.2, -0.5, 1 ).normalized();
    const double angle = 0.4;
    const auto [u, v] = a.perpendicular();
    // a third of one strip of the cone between the heights 5 and 5.5, and the planes of two degenerate triangles
    Vector<Vector3d> pts;
    Vector<Plane3d> planes = { Plane3d{}, Plane3d( Vector3d{}, 1 ) };
    for ( int i = 0; i < 10; ++i )
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

    // a wide cone with the points shifted along the normal by +0.1 and -0.1 in turn: the orthogonal fit of the generator is exact,
    // while the least squares line of the distance to the axis against the position along it is not
    const double wide = 1.2;
    Vector<Vector3d> widePts;
    for ( int i = 0; i < 12; ++i )
        for ( int j = 0; j < 8; ++j )
        {
            const double phi = 2 * PI * i / 12, t = 1 + 2.0 * j / 7;
            const Vector3d er = std::cos( phi ) * u + std::sin( phi ) * v;
            const Vector3d n = er * std::cos( wide ) - a * std::sin( wide );
            widePts.push_back( apex + t * ( a * std::cos( wide ) + er * std::sin( wide ) ) + ( ( i + j ) % 2 ? 0.1 : -0.1 ) * n );
        }
    const auto cone3 = findBestConeWithAxis( widePts, Line3d( apex, a ) );
    ASSERT_TRUE( cone3 );
    EXPECT_NEAR( ( cone3->apex() - apex ).length(), 0, 1e-9 );
    EXPECT_NEAR( cone3->angle, wide, 1e-9 );
}

TEST( MRMesh, FindBestTorus )
{
    std::mt19937 gen( 0 );
    // also with the axis along z far from the origin, where the coordinates of all points are nearly equal
    for ( const auto& [shift, dir] : { std::pair( Vector3d{}, Vector3d( 0.3, 0.4, 1 ) ), std::pair( Vector3d( 1e5, -1e5, 7e4 ), Vector3d( 0, 0, 1 ) ) } )
    {
        const Torus3d t( Vector3d( -1, 4, 2 ) + shift, dir.normalized(), 5, 1.5 );
        const auto [u, v] = t.axis.perpendicular();
        // a patch of the torus like a fillet: a quarter of the revolution and 2 radians of the tube
        Vector<Vector3d> pts;
        Vector<Line3d> normalLines;
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

        // with the points shifted along the normals by noise, the refined torus is closer to them than the true one
        Vector<Vector3d> noisyPts;
        for ( const auto& p : pts )
        {
            const double e = noise( gen, 0.05 );
            noisyPts.push_back( p + e * t.normal( p ) );
        }
        const auto cost = [&]( const Torus3d& x )
        {
            double sum = 0;
            for ( const auto& p : noisyPts )
                sum += sqr( x.distance( p ) );
            return sum;
        };
        const auto refined = refineTorus( noisyPts, initial );
        EXPECT_LT( cost( refined ), cost( t ) );
    }
}

} // namespace MR
