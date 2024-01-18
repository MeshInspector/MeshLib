#include "MRTriMath.h"
#include "MRGTest.h"

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

} //namespace MR
