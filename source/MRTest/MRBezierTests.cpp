#include <MRMesh/MRBezier.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, Bezier )
{
    const CubicBezierCurve3f bezier =
    {
        { 
            { 2.0f, 0.0f, 0.0f }, //p[0]
            { 1.0f, 0.0f, 0.0f }, //p[1]
            { 0.0f, 1.0f, 0.0f }, //p[2]
            { 0.0f, 2.0f, 0.0f }  //p[3]
        }
    };

    EXPECT_EQ( bezier.getPoint( 0.0f ), bezier.p[0] );
    EXPECT_EQ( bezier.getPoint( 0.25f ), Vector3f( 1.265625f, 0.171875f, 0.0f ) );
    EXPECT_EQ( bezier.getPoint( 0.5f ),  Vector3f( 0.625f,    0.625f,    0.0f ) );
    EXPECT_EQ( bezier.getPoint( 0.75f ), Vector3f( 0.171875f, 1.265625f, 0.0f ) );
    EXPECT_EQ( bezier.getPoint( 1.0f ), bezier.p[3] );
}

} // namespace MR
