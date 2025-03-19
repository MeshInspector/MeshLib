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

    Vector3f g;

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0.f, 1.f );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0.f, 1.f );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0.f, 1.f );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    std::optional<float> e;
    g = gradientInTri( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1.f, 1.f );
    EXPECT_NEAR( ( g - Vector3f{ 1, 1, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = findTriExitPos( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, -g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -1.f, -1.f );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );
    e = findTriExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -g );
    EXPECT_FALSE( e.has_value() );

    g = gradientInTri( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1.f, -1.f );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_NEAR( *e, 0.1f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, -1.f, -1.f );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, g );
    EXPECT_NEAR( *e, 0.9f, 1e-5f );

    g = gradientInTri( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1.f, -1.f );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = findTriExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = findTriExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -g );
    EXPECT_FALSE( e.has_value() );
}

} //namespace MR
