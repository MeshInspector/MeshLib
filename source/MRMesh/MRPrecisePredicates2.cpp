#include "MRPrecisePredicates2.h"
#include "MRHighPrecision.h"
#include "MRGTest.h"

namespace MR
{

bool ccw( const Vector2i & a, const Vector2i & b )
{
    if ( auto v = cross( Vector2ll{ a }, Vector2ll{ b } ) )
        return v > 0; // points are in general position

    // points 0, a, b are on the same line

    // permute points:
    // da.y >> da.x >> db.y >> db.x > 0

    if ( b.x )
        return b.x < 0;

    if ( b.y )
        return b.y > 0;

    if ( a.x )
        return a.x > 0;

    return a.y < 0;
}

bool ccw( const std::array<PreciseVertCoords2, 3> & vs )
{
    return ccw( vs.data() );
}

bool ccw( const PreciseVertCoords2* vs )
{
    bool odd = false;
    std::array<int, 3> order = {0, 1, 2};

    for ( int i = 0; i < 2; ++i )
    {
        for ( int j = i + 1; j < 3; ++j )
        {
            assert( vs[order[i]].id != vs[order[j]].id );
            if ( vs[order[i]].id > vs[order[j]].id )
            {
                odd = !odd;
                std::swap( order[i], order[j] );
            }
        }
    }

    return odd != ccw( vs[order[0]].pt, vs[order[1]].pt, vs[order[2]].pt );
}

SegmentSegmentIntersectResult doSegmentSegmentIntersect( const std::array<PreciseVertCoords2, 4> & vs )
{
    SegmentSegmentIntersectResult res;
    constexpr int a = 0;
    constexpr int b = 1;
    constexpr int c = 2;
    constexpr int d = 3;

    auto ccw = [&]( int p, int q, int r )
    {
        return MR::ccw( { vs[p], vs[q], vs[r] } );
    };

    const auto abc = ccw( a, b, c );
    res.cIsLeftFromAB = abc;
    const auto abd = ccw( a, b, d );
    if ( abc == abd )
        return res; // segment CD is located at one side of the line AB

    const auto cda = ccw( c, d, a );
    const auto cdb = ccw( c, d, b );
    if ( cda == cdb )
        return res; // segment AB is located at one side of the line CD

    res.doIntersect = true;
    return res;
}

Vector2i findSegmentSegmentIntersectionPrecise(
    const Vector2i& ai, const Vector2i& bi, const Vector2i& ci, const Vector2i& di )
{
    auto abc = cross( Vector2hp( ai - ci ), Vector2hp( bi - ci ) );
    if ( abc < 0 )
        abc = -abc;
    auto abd = cross( Vector2hp( ai - di ), Vector2hp( bi - di ) );
    if ( abd < 0 )
        abd = -abd;
    auto sum = abc + abd;
    if ( sum != HighPrecisionInt( 0 ) )
        return Vector2i{ Vector2d( abc * Vector2hp( di ) + abd * Vector2hp( ci ) ) / double( sum ) };
    auto adLSq = Vector2hp( di - ai ).lengthSq();
    auto bcLSq = Vector2hp( bi - ci ).lengthSq();
    if ( adLSq > bcLSq )
        return ci;
    else if ( bcLSq > adLSq )
        return di;
    else
        return Vector2i( Vector2d( Vector2hp( ai ) + Vector2hp( bi ) + Vector2hp( ci ) + Vector2hp( di ) ) * 0.5 );
}

Vector2f findSegmentSegmentIntersectionPrecise( 
    const Vector2f& a, const Vector2f& b, const Vector2f& c, const Vector2f& d,
    CoordinateConverters2 converters )
{
    auto ai{ converters.toInt( a ) };
    auto bi{ converters.toInt( b ) };
    auto ci{ converters.toInt( c ) };
    auto di{ converters.toInt( d ) };
    return converters.toFloat( findSegmentSegmentIntersectionPrecise( ai, bi, ci, di ) );
}

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
    std::array<PreciseVertCoords2, 7> vs =
    {
        PreciseVertCoords2{ 0_v, Vector2i{  0,  0 } },
        PreciseVertCoords2{ 1_v, Vector2i(  0,  0 ) },
        PreciseVertCoords2{ 2_v, Vector2i{  0,  1 } },
        PreciseVertCoords2{ 3_v, Vector2i{  0, -1 } },
        PreciseVertCoords2{ 4_v, Vector2i{  1,  0 } },
        PreciseVertCoords2{ 5_v, Vector2i{ -1,  0 } },
        PreciseVertCoords2{ 6_v, Vector2i{  0,  0 } },
    };

    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[2] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[3] } ) );
    EXPECT_TRUE(  ccw( { vs[0],vs[1],vs[4] } ) );
    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[5] } ) );
    EXPECT_FALSE( ccw( { vs[0],vs[1],vs[6] } ) );
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

} //namespace MR
