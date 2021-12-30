#include "MRViewportId.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, ViewportIterator )
{
    for ( [[maybe_unused]] ViewportId id : ViewportMask{0} )
        EXPECT_TRUE( false );

    int left = 3;
    for ( ViewportId id : ViewportMask{42} )
    {
        EXPECT_TRUE( id == ViewportId(2) || id == ViewportId(8) || id == ViewportId(32) );
        --left;
    }
    EXPECT_EQ( left, 0 );
}

} //namespace MR
