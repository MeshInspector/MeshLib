#include <MRMesh/MRMarkedContour.h>
#include <MRMesh/MRContour.h>
#include <MRMesh/MRVector3.h>
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, MarkedContour)
{
    auto mc = markedContour( Contour3f{ Vector3f{ 0, 0, 0 }, Vector3f{ 1, 0, 0 } } );
    auto rc = resample( mc, 2 );
    EXPECT_EQ( mc.contour, rc.contour );
    EXPECT_EQ( mc.marks, rc.marks );

    rc = resample( mc, 0.4f );
    EXPECT_EQ( rc.contour.size(), 4 );

    auto spline = makeSpline( rc );
    EXPECT_EQ( spline.contour.size(), 4 );
}

TEST(MRMesh, MakeClosedSpline)
{
    Contour3f c{
        Vector3f{ 0, 0, 0 },
        Vector3f{ 1, 0, 0 },
        Vector3f{ 1, 1, 0 },
        Vector3f{ 0, 1, 0 },
        Vector3f{ 0, 0, 0 }
    };
    SplineSettings s{ .samplingStep = 0.4f };
    auto spline = makeSpline( c, s );
    EXPECT_EQ( spline.contour.size(), 13 );
    EXPECT_EQ( spline.contour.front(), spline.contour.back() );
    EXPECT_EQ( spline.marks.count(), 5 );
}

} //namespace MR
