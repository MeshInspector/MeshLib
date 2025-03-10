#include "MRMesh/MRGTest.h"
#include "MRMesh/MRIsNaN.h"

namespace MR
{

// global variables with external visibility to avoid compile-time optimizations
float gTestNaN = cQuietNan;
float gTestZero = 0;

TEST( MRMesh, NaN )
{
    // tests basic precondition for Marching Cubes algorithm to be correct
    EXPECT_FALSE( gTestNaN < gTestZero || gTestNaN >= gTestZero );
}

} //namespace MR
