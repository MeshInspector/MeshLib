#include <MRMesh/MRPrecisePredicates2.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRGTest.h>
#include <algorithm>
#include <climits>

namespace MR
{

TEST( MRMesh, doSegmentSegmentIntersect )
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

TEST( MRMesh, sosCCW )
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

TEST( MRMesh, sosCCW2 )
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

TEST( MRMesh, sosSmaller2 )
{
    std::array<PreciseVertCoords2, 5> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{ 0, 0 } },
        PreciseVertCoords2{ 1_v, Vector2i( 1, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i{ 0, 1 } },
        PreciseVertCoords2{ 3_v, Vector2i{ 0, 2 } },
        PreciseVertCoords2{ 4_v, Vector2i{ 0, 1 } } // vs[4].pt == vs[2].pt
    };

    // not-degenerate
    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[2], vs[3] } ) );
    EXPECT_FALSE( smaller2( { vs[0], vs[1], vs[3], vs[2] } ) );
    EXPECT_FALSE( smaller2( { vs[1], vs[0], vs[2], vs[3] } ) );

    // partially degenerate
    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[4], vs[2] } ) );
    EXPECT_FALSE( smaller2( { vs[0], vs[1], vs[2], vs[4] } ) );
    EXPECT_FALSE( smaller2( { vs[1], vs[0], vs[4], vs[2] } ) );
}

TEST( MRMesh, sosSmaller2FullDegen )
{
    std::array<PreciseVertCoords2, 4> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 0, 0 ) }
    };

    EXPECT_TRUE(  smaller2( { vs[0], vs[1], vs[2], vs[3] } ) );

    // test that maximum degree in smaller2 can cope with most degenerate situation possible
    do
    {
        (void)smaller2( { vs[0], vs[1], vs[2], vs[3] } );
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, sosInCircle )
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

TEST( MRMesh, segmentIntersectionOrder2a )
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

    // intersection of 45 and 03 is closer to 4 than intersection of 45 and 12
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[3], vs[1], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[5], vs[4], vs[0], vs[3], vs[1], vs[2] } ) );

    // intersection of 45 and 03 is closer to 4 than intersection of 45 and 02
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[3], vs[0], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[4], vs[5], vs[0], vs[2], vs[0], vs[3] } ) );
}

TEST( MRMesh, segmentIntersectionOrder2FullDegen )
{
    std::array<PreciseVertCoords2, 6> vs = 
    { 
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 2_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 4_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 5_v, Vector2i( 0, 0 ) }
    };

    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ).doIntersect );
    EXPECT_TRUE(  doSegmentSegmentIntersect( { vs[0], vs[2], vs[1], vs[3] } ).doIntersect );
    EXPECT_FALSE( doSegmentSegmentIntersect( { vs[0], vs[3], vs[1], vs[2] } ).doIntersect );

    // test that maximum degree in segmentIntersectionOrder can cope with most degenerate situation possible
    do
    {
        if ( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ).doIntersect
            && doSegmentSegmentIntersect( { vs[0], vs[1], vs[4], vs[5] } ).doIntersect )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, doSegmentSegmentIntersectPartialDegen )
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

TEST( MRMesh, sosInCircle2 )
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

TEST( MRMesh, segmentIntersectionOrder2b )
{
    PreciseVertCoords2 vs[6] =
    {
        // s:
        PreciseVertCoords2{ 0_v, Vector2i( 0, 0 ) },
        PreciseVertCoords2{ 1_v, Vector2i( 3, 0 ) },
        // sa:
        PreciseVertCoords2{ 2_v, Vector2i( 1,-1 ) },
        PreciseVertCoords2{ 3_v, Vector2i( 1, 1 ) },
        // sb:
        PreciseVertCoords2{ 5_v, Vector2i( 2,-1 ) },
        PreciseVertCoords2{ 6_v, Vector2i( 2, 1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[5], vs[4] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[1], vs[0], vs[3], vs[2], vs[5], vs[4] } ) );

    // swapped sa and sb
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[3], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[4], vs[3], vs[2] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[1], vs[0], vs[5], vs[4], vs[3], vs[2] } ) );

    // shared point in sa and sb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[2], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[2], vs[5] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[5], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[2], vs[5] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[4], vs[5], vs[5], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[4], vs[5], vs[2] } ) );
}

TEST( MRMesh, findTwoSegmentsIntersection )
{
    const Vector3i a( -100, -50 , 0 );
    const Vector3i b(  300, 150 , 0 );

    auto v = findTwoSegmentsIntersection( a, b, { 0, -200, 0 }, { 0, 100, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, -100, 0 }, { 0, 100, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, -100, 0 }, { 0, 300, 0 } );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, Vector3i{} );

    v = findTwoSegmentsIntersection( a, b, { 0, 100, 0 }, { 0, 300, 0 } );
    EXPECT_FALSE( v.has_value() );

    // test with largest possible values
    constexpr int h = INT_MAX / 2 - 3;
    const Vector3i d( 1, 2, 3 );
    v = findTwoSegmentsIntersection( Vector3i{ h,  h, h } + d, Vector3i{ -h, -h, -h } + d,
                                     Vector3i{ h, -h, h } + d, Vector3i{ -h,  h, -h } + d );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, d );

    v = findTwoSegmentsIntersection( Vector3i{ h,  h, h } + d, Vector3i{ -h, -h, -h } + d,
                                     Vector3i{-h,  h,-h } + d, Vector3i{  h, -h,  h } + d );
    EXPECT_TRUE( v.has_value() );
    EXPECT_EQ( *v, d );
}

TEST( MRMesh, orientParaboloid3d )
{
    // large numbers requiring more than 64-bit arithmetic, and degeneration (b==c)
    const Vector2i a{ 54209929, -710917541 };
    const Vector2i b{ 0, -365379885 };
    EXPECT_FALSE( orientParaboloid3d( a, b, b ) );
}

TEST( MRMesh, doTriangleSegmentIntersect )
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

TEST( MRMesh, doTriangleSegmentIntersect2 )
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

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 123
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[1], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[7], vs[6], vs[0], vs[4], vs[5], vs[1], vs[2], vs[3] } ) );

    // one shared point 0: triangles 045 and 023 do not intersect one another
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[0], vs[2], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[2], vs[3], vs[0], vs[4], vs[5] } ) );

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 143 (one shared point 4)
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[1], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[1], vs[4], vs[3], vs[0], vs[4], vs[5] } ) );

    // intersection of 67 and 045 is closer to 6 than intersection of 67 and 043 (two shared points 0 and 4)
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[5], vs[0], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[6], vs[7], vs[0], vs[4], vs[3], vs[0], vs[4], vs[5] } ) );
}

TEST( MRMesh, doTriangleSegmentIntersectFullDegen )
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

TEST( MRMesh, segmentIntersectionOrder3FullDegen )
{
    std::array<PreciseVertCoords, 8> vs;
    for ( VertId i = 0_v; i < 8; ++i )
        vs[i].id = i; //and point coordinate is (0,0,0)

    // test that maximum degree in segmentIntersectionOrder can cope with most degenerate situation possible

    // no shared vertices
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } )
         && doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end(), []( const auto & l, const auto & r ) { return l.id < r.id; } ) );

    // one shared vertex
    do
    {
        if( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } )
         && doTriangleSegmentIntersect( { vs[5], vs[6], vs[2], vs[0], vs[1] } ) )
        {
            (void)segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[2] } );
        }
    }
    while ( std::next_permutation( vs.begin(), vs.end() - 1, []( const auto & l, const auto & r ) { return l.id < r.id; } ) );
}

TEST( MRMesh, segmentIntersectionOrder3a )
{
    PreciseVertCoords vs[8] =
    {
        // s:
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 3, 0, 0 ) },
        // ta:
        PreciseVertCoords{ 2_v, Vector3i( 1,-1,-1 ) },
        PreciseVertCoords{ 3_v, Vector3i( 1, 1,-1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, 0, 1 ) },
        // tb:
        PreciseVertCoords{ 5_v, Vector3i( 2,-1,-1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2, 1,-1 ) },
        PreciseVertCoords{ 7_v, Vector3i( 2, 0, 1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[5], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[3], vs[2], vs[4], vs[6], vs[5], vs[7] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[1], vs[0], vs[3], vs[2], vs[4], vs[6], vs[5], vs[7] } ) );

    // swapped ta and tb
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[3], vs[4] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[7], vs[2], vs[4], vs[3] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[7], vs[6], vs[2], vs[4], vs[3] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[1], vs[0], vs[5], vs[7], vs[6], vs[2], vs[4], vs[3] } ) );

    // one shared point in ta and tb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[6], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[4] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[6], vs[4], vs[5], vs[6], vs[7] } ) );

    // two shared points in ta and tb
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[2], vs[3], vs[7] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[4], vs[3] } ) );
    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[6], vs[7], vs[5], vs[6], vs[7] } ) );
}

TEST( MRMesh, segmentIntersectionOrder3b )
{
    PreciseVertCoords vs[8] =
    {
        // s:
        PreciseVertCoords{ 0_v, Vector3i( 0, 0, 0 ) },
        PreciseVertCoords{ 1_v, Vector3i( 4, 0, 0 ) },
        // shared vertex of tris:
        PreciseVertCoords{ 2_v, Vector3i( 2, 0, 1 ) },
        // ta:
        PreciseVertCoords{ 3_v, Vector3i( 1,  100, -1 ) },
        PreciseVertCoords{ 4_v, Vector3i( 1, -100, -1 ) },
        // tb:
        PreciseVertCoords{ 5_v, Vector3i( 3,    1, -1 ) },
        PreciseVertCoords{ 6_v, Vector3i( 2,   -1, -1 ) }
    };

    EXPECT_TRUE(  segmentIntersectionOrder( { vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[2] } ) );
    EXPECT_FALSE( segmentIntersectionOrder( { vs[0], vs[1], vs[5], vs[6], vs[2], vs[2], vs[3], vs[4] } ) );
}

TEST( MRMesh, getToIntConverter )
{
    auto toInt = getToIntConverter( Box3d( {0,0,-1.0}, {0,0,1.0} ) );
    auto i0 = toInt( { 0,0,-1.f } );
    auto i1 = toInt( { 0,0, 1.f } );
    // check that sum and difference of any two points can be computed in integer without overflow
    EXPECT_LE( -i0.z, INT_MAX / 2 );
    EXPECT_LE(  i1.z, INT_MAX / 2 );
    EXPECT_GT( i1.z - i0.z, 0 );
}

} //namespace MR
