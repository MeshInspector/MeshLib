#include "MRContour.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, calcOrientedArea )
{
    Contour2f contour = {
        { 0, 0 },
        { 1, 0 },
        { 0, 1 },
        { 0, 0 }
    };
    auto area = calcOrientedArea( contour );
    EXPECT_NEAR( area, -0.5f, 1e-6f );
}

} //namespace MR
