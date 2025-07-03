#include <MRMesh/MRPreciseSegmentIntersectionOrder3.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, PreciseSegmentIntersectionOrder3 )
{
    PreciseVertCoords segm[2] =
    {
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 3, 0, 0 ) }
    };

    PreciseVertCoords ta[3] =
    {
        PreciseVertCoords{ 2_v, Vector3i( 1,-1,-1 ) },
        PreciseVertCoords{ 3_v, Vector3i( 1, 1,-1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, 0, 1 ) }
    };

    PreciseVertCoords tb[3] =
    {
        PreciseVertCoords{ 5_v, Vector3i( 2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 2, 0, 1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( segm, ta, tb ) );
    EXPECT_FALSE( segmentIntersectionOrder( segm, tb, ta ) );

    std::swap( ta[0], ta[1] );
    EXPECT_TRUE(  segmentIntersectionOrder( segm, ta, tb ) );
    EXPECT_FALSE( segmentIntersectionOrder( segm, tb, ta ) );

    std::swap( tb[0], tb[1] );
    EXPECT_TRUE(  segmentIntersectionOrder( segm, ta, tb ) );
    EXPECT_FALSE( segmentIntersectionOrder( segm, tb, ta ) );

    std::swap( segm[0], segm[1] );
    EXPECT_FALSE( segmentIntersectionOrder( segm, ta, tb ) );
    EXPECT_TRUE(  segmentIntersectionOrder( segm, tb, ta ) );
}

} //namespace MR
