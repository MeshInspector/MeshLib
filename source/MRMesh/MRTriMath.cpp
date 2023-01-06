#include "MRTriMath.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, posFromTriEdgeLengths)
{
    EXPECT_EQ( posFromTriEdgeLengths( 4., 5., 3. ), Vector2d( 4., 0. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 5., 4., 3. ), Vector2d( 4., 3. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 5., 4., 10. ), std::nullopt );
    EXPECT_EQ( posFromTriEdgeLengths( 1., 1., 0. ), Vector2d( 1., 0. ) );
    EXPECT_EQ( posFromTriEdgeLengths( 1., 2., 0. ), std::nullopt );
}

} //namespace MR
