#include <MRMesh/MRTriMath.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, TriMath )
{
    EXPECT_EQ( circumcircleCenter( Vector3d{ 0, 0, 0 }, Vector3d{ 1, 0, 0 }, Vector3d{ 0, 1, 0 } ), Vector3d( 0.5, 0.5, 0 ) );
    EXPECT_EQ( circumcircleCenter( Vector3d{ 0, 0, 1 }, Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 0 } ), Vector3d( 0, 0, 0.5 ) );
    EXPECT_EQ( circumcircleCenter( Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 1 }, Vector3d{ 0, 0, 0 } ), Vector3d( 0, 0, 0.5 ) );
    EXPECT_EQ( circumcircleCenter( Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 1 } ), Vector3d( 0, 0, 0.5 ) );

    Vector3d centerPos, centerNeg;
    EXPECT_FALSE( circumballCenters( Vector3d{ 0, 0, 0 }, Vector3d{ 1, 0, 0 }, Vector3d{ 0, 1, 0 }, 0.1, centerPos, centerNeg ) );
    EXPECT_TRUE(  circumballCenters( Vector3d{ 0, 0, 0 }, Vector3d{ 2, 0, 0 }, Vector3d{ 0, 2, 0 }, std::sqrt( 3.0 ), centerPos, centerNeg ) );
    EXPECT_NEAR( ( centerPos - Vector3d( 1, 1,  1 ) ).length(), 0.0, 1e-15 );
    EXPECT_NEAR( ( centerNeg - Vector3d( 1, 1, -1 ) ).length(), 0.0, 1e-15 );

    EXPECT_EQ( posFromTriEdgeLengths( 4., 5., 3. ), Vector2d( 4., 0. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 5., 4., 3. ), Vector2d( 4., 3. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 5., 4., 10. ), std::nullopt );
    EXPECT_EQ( posFromTriEdgeLengths( 1., 1., 0. ), Vector2d( 1., 0. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 1., 2., 0. ), std::nullopt );

    EXPECT_EQ( quadrangleOtherDiagonal( 1., 1., 1., 1., 1. ), std::sqrt( 3. ) );
    EXPECT_EQ( quadrangleOtherDiagonal( 4., 5., 3., 4., 5. ), 8. );
    EXPECT_EQ( quadrangleOtherDiagonal( 5., 4., 3., 5., 4. ), 8. );
    EXPECT_EQ( quadrangleOtherDiagonal( 6., 4., 3., 5., 4. ), std::nullopt );
    EXPECT_EQ( quadrangleOtherDiagonal( 5., 4., 3., 4., 5. ), std::sqrt( 73. ) );
}

TEST( MRMesh, gradientInTri )
{
    Vector3d a{ 1, 1, 1 };
    Vector3d b{ 1, 2, 1 };
    Vector3d c{ 1, 1, 2 };
    EXPECT_EQ( gradientInTri( a, b, c, 1., 1., 1. ), Vector3d( 0, 0, 0 ) );
    EXPECT_EQ( gradientInTri( a, b, c, 1., 3., 1. ), Vector3d( 0, 2, 0 ) );
    EXPECT_EQ( gradientInTri( a, b, c, 1., 1., 3. ), Vector3d( 0, 0, 2 ) );

    std::optional<Vector3f >g;
    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 1, 0, 0 }, 0.f, 1.f );
    EXPECT_FALSE( g.has_value() );

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0.f, 1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0.f, 1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0.f, 1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    std::optional<float> e;
    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1.f, 1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ 1, 1, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, *g );
    EXPECT_FALSE( e.has_value() );
    e = findTriExitPos( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, -*g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -1.f, -1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, *g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );
    e = findTriExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -*g );
    EXPECT_FALSE( e.has_value() );

    g = gradientInTri( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1.f, -1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, *g );
    EXPECT_NEAR( *e, 0.1f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, -1.f, -1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, *g );
    EXPECT_NEAR( *e, 0.9f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1.f, -1.f );
    EXPECT_TRUE( g.has_value() );
    EXPECT_NEAR( ( *g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, *g );
    EXPECT_FALSE( e.has_value() );
    e = findTriExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -*g );
    EXPECT_FALSE( e.has_value() );
}

TEST( MRMesh, tangentPlaneToSpheres )
{
    auto p0 = tangentPlaneToSpheres( Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 0 }, Vector3d{ 0, 0, 0 }, 1., 2., 3. );
    EXPECT_FALSE( p0.has_value() );

    auto p1 = tangentPlaneToSpheres( Vector3d{ 0, 0, 0 }, Vector3d{ 0.1, 0, 0 }, Vector3d{ 0.2, 0, 0 }, 1., 2., 3. );
    EXPECT_FALSE( p1.has_value() );

    auto p2 = tangentPlaneToSpheres( Vector3d{ 0, 0, -1 }, Vector3d{ 5, 0, -1 }, Vector3d{ 0, 5, -1 }, 1., 1., 1. );
    EXPECT_TRUE( p2.has_value() );
    EXPECT_NEAR( distance( p2->n, Vector3d{ 0, 0, 1 } ), 0, 1e-15 );
    EXPECT_NEAR( p2->d, 0, 1e-15 );

    auto p3 = tangentPlaneToSpheres( Vector3d{ 0, 0, -1 }, Vector3d{ 5, 0, -2 }, Vector3d{ 0, 5, -3 }, 1., 2., 3. );
    EXPECT_TRUE( p3.has_value() );
    EXPECT_NEAR( distance( p3->n, Vector3d{ 0, 0, 1 } ), 0, 1e-15 );
    EXPECT_NEAR( p3->d, 0, 1e-15 );
}

TEST( MRMesh, quadrangleOtherDiagonal )
{
    const double exp1 = 3.5551215012835908;
    auto d1 = quadrangleOtherDiagonal( 3., 4., 6., 3., 4. );
    EXPECT_TRUE( d1.has_value() );
    EXPECT_NEAR( exp1, *d1, 1e-15 );
    auto d1f = quadrangleOtherDiagonal( 3.f, 4.f, 6.f, 3.f, 4.f );
    EXPECT_TRUE( d1f.has_value() );
    EXPECT_NEAR( float( exp1 ), *d1f, 1e-6f );

    auto d2 = quadrangleOtherDiagonal( 4., 6., 3., 4., 6. );
    EXPECT_FALSE( d2.has_value() );
    auto d2f = quadrangleOtherDiagonal( 4.f, 6.f, 3.f, 4.f, 6.f );
    EXPECT_FALSE( d2f.has_value() );

    const double exp3 = 4.3870982579476117;
    auto d3 = quadrangleOtherDiagonal( 3., 4., 6., 3.5, 4.5 );
    EXPECT_TRUE( d3.has_value() );
    EXPECT_NEAR( exp3, *d3, 1e-15 );
    auto d3f = quadrangleOtherDiagonal( 3.f, 4.f, 6.f, 3.5f, 4.5f );
    EXPECT_TRUE( d3f.has_value() );
    EXPECT_NEAR( float( exp3 ), *d3f, 1e-6f );
}

TEST( MRMesh, triangleAnglesFromEdgeLengths )
{
    EXPECT_NEAR(            cotan( 3., 4., 5. ), 4. / 3, 1e-15 );
    EXPECT_NEAR( tanSqOfHalfAngle( 3., 4., 5. ), 1. / 9, 1e-15 );

    EXPECT_NEAR(            cotan( 4., 5., 3. ), 3. / 4, 1e-15 );
    EXPECT_NEAR( tanSqOfHalfAngle( 4., 5., 3. ), 1. / 4, 1e-15 );

    EXPECT_NEAR(            cotan( 5., 3., 4. ), 0., 1e-15 );
    EXPECT_NEAR( tanSqOfHalfAngle( 5., 3., 4. ), 1., 1e-15 );

    EXPECT_NEAR(            cotan( 3.f, 4.f, 5.f ), 4.f / 3, 1e-6f );
    EXPECT_NEAR( tanSqOfHalfAngle( 3.f, 4.f, 5.f ), 1.f / 9, 1e-6f );

    EXPECT_NEAR(            cotan( 4.f, 5.f, 3.f ), 3.f / 4, 1e-6f );
    EXPECT_NEAR( tanSqOfHalfAngle( 4.f, 5.f, 3.f ), 1.f / 4, 1e-6f );

    EXPECT_NEAR(            cotan( 5.f, 3.f, 4.f ), 0.f, 1e-6f );
    EXPECT_NEAR( tanSqOfHalfAngle( 5.f, 3.f, 4.f ), 1.f, 1e-6f );
}

} //namespace MR
