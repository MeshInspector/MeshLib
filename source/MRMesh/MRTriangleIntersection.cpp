#include "MRTriangleIntersection.h"
#include "MRGTest.h"

namespace MR
{
TEST( MRMesh, TriangleSegmentIntersectFloat )
{
    Vector3f a{2,  1, 0};
    Vector3f b{-2,  1, 0};
    Vector3f c{0, -2, 0};

    Vector3f d{0, 0, -1};
    Vector3f e{0, 0,  1};

    bool intersection = doTriangleSegmentIntersect( a, b, c, d, e );

    EXPECT_TRUE( intersection );
}
}
