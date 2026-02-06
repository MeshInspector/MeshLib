#include <MRMesh/MRContour.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, calcOrientedArea )
{
    Contour2f contour2 = {
        { 0, 0 },
        { 1, 0 },
        { 0, 1 },
        { 0, 0 }
    };
    auto area2 = calcOrientedArea( contour2 );
    EXPECT_NEAR( area2, -0.5f, 1e-6f );

    auto area2d = calcOrientedArea<float, double>( contour2 );
    EXPECT_NEAR( area2d, -0.5, 1e-12 );

    Contour3f contour3 = {
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 0 }
    };
    auto area3 = calcOrientedArea( contour3 );
    EXPECT_NEAR( area3.length(), 0.5f, 1e-6f );
    EXPECT_NEAR( area3.z, 0.5f, 1e-6f );

    auto area3d = calcOrientedArea<float, double>( contour3 );
    EXPECT_NEAR( area3d.length(), 0.5, 1e-12 );
    EXPECT_NEAR( area3d.z, 0.5, 1e-12 );
}

TEST( MRMesh, calcLength )
{
    Contour2f contour2 = {
        { 0, 0 },
        { 1, 0 },
        { 1, 1 },
        { 0, 1 },
        { 0, 0 }
    };
    auto length2 = calcLength( contour2 );
    EXPECT_NEAR( length2, 4.0f, 1e-6f );

    EXPECT_EQ( 0, findContourPointByLength( contour2, -1 ) );
    EXPECT_EQ( 0, findContourPointByLength( contour2, 0.1f ) );
    EXPECT_EQ( 2, findContourPointByLength( contour2, 1.7f ) );
    EXPECT_EQ( 4, findContourPointByLength( contour2, 3.9f ) );
    EXPECT_EQ( 4, findContourPointByLength( contour2, 5 ) );

    auto length2d = calcLength<Vector2f, double>( contour2 );
    EXPECT_NEAR( length2d, 4.0, 1e-12 );

    Contour3f contour3 = {
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 1, 1, 0 },
        { 0, 1, 0 },
        { 0, 0, 0 }
    };
    auto length3 = calcLength( contour3 );
    EXPECT_NEAR( length3, 4.0f, 1e-6f );

    auto length3d = calcLength<Vector3f, double>( contour3 );
    EXPECT_NEAR( length3d, 4.0, 1e-12 );
}

} //namespace MR
