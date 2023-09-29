#include "MRId.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, Id) 
{
    EdgeId e(1);
    FaceId f(2);
    VertId v(3);

    int ei = e;
    int fi = f;
    int vi = v;

    EXPECT_EQ( e, ei );
    EXPECT_EQ( f, fi );
    EXPECT_EQ( v, vi );

    EXPECT_EQ( EdgeId().valid(), false );
    EXPECT_EQ( e.valid(), true );
}

} //namespace MR
