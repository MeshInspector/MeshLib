#include "MRMesh/MRBestFit.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRGTest.h"

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

} // namespace MR
