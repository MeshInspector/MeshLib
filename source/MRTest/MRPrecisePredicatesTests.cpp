#include <MRMesh/MRPrecisePredicates2.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <MRMesh/MRGTest.h>
#include <algorithm>

namespace MR
{

TEST( MRMesh, PrecisePredicates2 )
{
    std::array<PreciseVertCoords2, 4> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( -1,  0 ) }, //a
        PreciseVertCoords2{ 1_v, Vector2i{  1,  0 } }, //b

        PreciseVertCoords2{ 2_v, Vector2i(  0,  1 ) }, //c
        PreciseVertCoords2{ 3_v, Vector2i{  0, -1 } }  //d
    };

    auto res = doSegmentSegmentIntersect( vs );
    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( res.cIsLeftFromAB );

    std::swap( vs[2], vs[3] );
    res = doSegmentSegmentIntersect( vs );
    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( !res.cIsLeftFromAB );

    vs[3].pt.y = -5;
    res = doSegmentSegmentIntersect( vs );
    EXPECT_FALSE( res.doIntersect );
}

TEST( MRMesh, PrecisePredicates2other )
{
    std::array<PreciseVertCoords2, 9> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{  0,  0 } },
        PreciseVertCoords2{ 1_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 2_v, Vector2i{  0,  1 } },
        PreciseVertCoords2{ 3_v, Vector2i{  0, -1 } },
        PreciseVertCoords2{ 4_v, Vector2i{  1,  0 } },
        PreciseVertCoords2{ 5_v, Vector2i{ -1,  0 } },
        PreciseVertCoords2{ 6_v, Vector2i{  0,  0 } },
        PreciseVertCoords2{ 7_v, Vector2i{  0,  1 } },
        PreciseVertCoords2{ 8_v, Vector2i{  0, -1 } }
    };

    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[2] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[3] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[4] } ) );
    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[5] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[6] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[2],vs[7] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[3],vs[8] } ) );
}

TEST( MRMesh, PrecisePredicates2more )
{
    std::array<PreciseVertCoords2, 4> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ 1, 0 } },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 1 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ 0, 1 } },
        PreciseVertCoords2{ 3_v, Vector2i{ 1, 0 } }
    };

    EXPECT_FALSE( ccw( { vs[1],vs[0],vs[2] } ) );
    EXPECT_TRUE(  ccw( { vs[2],vs[3],vs[0] } ) );
}

TEST( MRMesh, PrecisePredicates2InCircle )
{
    std::array<PreciseVertCoords2, 4> vs =
    {
        PreciseVertCoords2{ 3_v, Vector2i{ -1, 2 } },
        PreciseVertCoords2{ 2_v, Vector2i( 0 , 0 ) },
        PreciseVertCoords2{ 0_v, Vector2i{ 3, 10 } },
        PreciseVertCoords2{ 1_v, Vector2i{ 0 , 0 } }
    };
    EXPECT_TRUE( ccw( { vs[0],vs[1],vs[2] } ) );

    // These 3 proves that vs[3] is inside vs[0]vs[1]vs[2] triangle
    EXPECT_TRUE( ccw( { vs[0],vs[1],vs[3] } ) );
    EXPECT_TRUE( ccw( { vs[1],vs[2],vs[3] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[0],vs[3] } ) );

    // Check that vs[3] is inCircle
    EXPECT_TRUE( inCircle( vs ) );
}

TEST( MRMesh, PrecisePredicates2More )
{
    const std::array<PreciseVertCoords2, 6> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i(  0, -1 ) },
        PreciseVertCoords2{ 1_v, Vector2i(  0, -1 ) },

        PreciseVertCoords2{ 2_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 3_v, Vector2i(  0,  0 ) },

        PreciseVertCoords2{ 4_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 5_v, Vector2i(  1,  0 ) }
    };

    // both segments 03 and 12 intersect line segment 45
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[4], vs[5] } ).doIntersect );
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[1], vs[2], vs[4], vs[5] } ).doIntersect );

    // segments 03 and 12 intersect one with another
    EXPECT_TRUE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[1], vs[2] } ).doIntersect );
}

TEST( MRMesh, PrecisePredicates2FullDegen )
{
    std::array<PreciseVertCoords2, 4> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 0, 0 ) }
    };

    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ).doIntersect );
    EXPECT_TRUE(  doSegmentSegmentIntersect( { vs[0], vs[2], vs[1], vs[3] } ).doIntersect );
    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[1], vs[2] } ).doIntersect );
}

TEST( MRMesh, PrecisePredicates2PartialDegen )
{
    EXPECT_TRUE( doSegmentSegmentIntersect( {
        PreciseVertCoords2{ 0_v, { 0,  0} },
        PreciseVertCoords2{ 2_v, {-1, -1} },
        PreciseVertCoords2{ 1_v, {-1, -1} },
        PreciseVertCoords2{ 3_v, {-1, -1} } } ).doIntersect );

    EXPECT_TRUE( doSegmentSegmentIntersect( {
        PreciseVertCoords2{ 1_v, { 0,  0} },
        PreciseVertCoords2{ 2_v, {-2, -2} },
        PreciseVertCoords2{ 0_v, {-1, -1} },
        PreciseVertCoords2{ 3_v, {-1, -1} } } ).doIntersect );

    // degenerated segment with ends at last vertices (with smallest perturbation) never intersects anything
    for ( int x = -1; x <= 1; ++x )
        for ( int y = -1; y <= 1; ++y )
        {
            if ( x == 0 && y == 0 )
                continue;

            const PreciseVertCoords2 p2{ 2_v, {x, y} };
            const PreciseVertCoords2 p3{ 3_v, {x, y} };

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 0_v, {0, 0} },
                PreciseVertCoords2{ 1_v, {x, y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 1_v, {0, 0} },
                PreciseVertCoords2{ 0_v, {x, y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 0_v, {0, 0} },
                PreciseVertCoords2{ 1_v, {2*x, 2*y} }, p2, p3 } ).doIntersect );

            EXPECT_FALSE( doSegmentSegmentIntersect( {
                PreciseVertCoords2{ 1_v, {0, 0} },
                PreciseVertCoords2{ 0_v, {2*x, 2*y} }, p2, p3 } ).doIntersect );
        }
}

TEST( MRMesh, PrecisePredicates2InCircle2 )
{
    std::array<PreciseVertCoords2, 5> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ -106280744 , -1002263723 } },
        PreciseVertCoords2{ 1_v, Vector2i( -187288916 , -172107608 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ -25334363 , -1063004405 } },
        PreciseVertCoords2{ 3_v, Vector2i{ -15200618 , -10122159 } },
        PreciseVertCoords2{ 4_v, Vector2i{ -106280744 , -1002263723 } }
    };

    // Prove that 0_v 2_v 4_v circle is in +Y half plane (4_v 2_v is horde in lower part)
    EXPECT_FALSE( ccw( { vs[2],vs[4],vs[3] } ) ); // 3_v is to the right of 2-4 vec
    
    EXPECT_FALSE( inCircle( { vs[4],vs[2],vs[0],vs[3] } ) ); // 3_v is in circle

    // prove that 0_v is inside 142 triangle
    EXPECT_TRUE( ccw( { vs[1],vs[4],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[4],vs[2],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[1],vs[0] } ) );
    // it means that 142 circle should be larger in +Y half plane and so 3_v should be inside it
    EXPECT_FALSE( inCircle( { vs[1],vs[4],vs[2],vs[3] } ) );
}

TEST( MRMesh, PrecisePredicates3 )
{
    const std::array<PreciseVertCoords, 5> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i(  2,  1, 0 ) }, //a
        PreciseVertCoords{ 1_v, Vector3i{ -2,  1, 0 } }, //b
        PreciseVertCoords{ 2_v, Vector3i{  0, -2, 0 } }, //c

        PreciseVertCoords{ 3_v, Vector3i{  0, 0, -1 } }, //d
        PreciseVertCoords{ 4_v, Vector3i{  0, 0,  1 } }  //e
    };

    auto res = doTriangleSegmentIntersect( vs );

    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( res.dIsLeftFromABC );
}

TEST( MRMesh, PrecisePredicates3More )
{
    const std::array<PreciseVertCoords, 8> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i(  0, -1, -1 ) },
        PreciseVertCoords{ 1_v, Vector3i(  0, -1, -1 ) },

        PreciseVertCoords{ 2_v, Vector3i{  0, -1,  1 } },

        PreciseVertCoords{ 3_v, Vector3i{  0,  1,  1 } },

        PreciseVertCoords{ 4_v, Vector3i{  0, -1,  1 } },

        PreciseVertCoords{ 5_v, Vector3i{  0,  1,  1 } },

        PreciseVertCoords{ 6_v, Vector3i{  0,  0,  1 } },
        PreciseVertCoords{ 7_v, Vector3i{  1,  0,  1 } }
    };

    // both triangles 045 and 123 intersect line segment 67
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[6], vs[7] } ).doIntersect );
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[1], vs[2], vs[3], vs[6], vs[7] } ).doIntersect );

    // triangles 045 and 123 intersect one with another
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[1], vs[2] } ).doIntersect );
    EXPECT_TRUE( doTriangleSegmentIntersect( { vs[0], vs[4], vs[5], vs[3], vs[1] } ).doIntersect );
}

TEST( MRMesh, PrecisePredicates3FullDegen )
{
    std::array<PreciseVertCoords, 5> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 2_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 3_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 4_v, Vector3i( 0, 0, 0 ) }
    };

    do
    {
        if ( vs[0].id < vs[1].id && vs[1].id < vs[2].id && vs[3].id < vs[4].id ) // ignore same triangles and segments with changed order of vertices
        {
            //spdlog::info( "{}{}{}x{}{}: {}", (int)vs[0].id, (int)vs[1].id, (int)vs[2].id, (int)vs[3].id, (int)vs[4].id,
            //    doTriangleSegmentIntersect( { vs[0], vs[1], vs[2], vs[3], vs[4] } ).doIntersect );
            EXPECT_EQ( doTriangleSegmentIntersect( { vs[0], vs[1], vs[2], vs[3], vs[4] } ).doIntersect, vs[3].id == 1 && vs[4].id == 3 );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

} //namespace MR
