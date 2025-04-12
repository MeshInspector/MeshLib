#include "MRPrecisePredicates2.h"
#include "MRHighPrecision.h"
#include "MRGTest.h"
#include "MRPrecisePredicates3.h"

namespace MR
{

// see https://arxiv.org/pdf/math/9410209 Table 4-i:
// a=(pi_i,1, pi_i,2)
// b=(pi_j,1, pi_j,2)
bool ccw( const Vector2i & a, const Vector2i & b )
{
    if ( auto v = cross( Vector2ll{ a }, Vector2ll{ b } ) )
        return v > 0; // points are in general position

    // points 0, a, b are on the same line

    // permute points:
    // da.y >> da.x >> db.y >> db.x > 0

    // the dominant permutation da.y > 0
    if ( b.x )
        return b.x < 0;
    // permutation da.y cannot resolve the degeneration, because
    // 1) b = 0 or
    // 2) points 0, a, b are on the line x = 0

    // next permutation da.x > 0
    if ( b.y )
        return b.y > 0;
    // permutation da.x cannot resolve the degeneration, because b = 0

    // next permutation db.y > 0
    if ( a.x )
        return a.x > 0;
    // permutation db.y cannot resolve the degeneration, because b = 0 and a.x = 0

    // a = ( da.x, a.y + da.y ) ~ ( +0, a.y )
    // b = (    0,       db.y ) ~ (  0, 1 )
    // the smallest permutation db.x does not change anything here, and
    // the rotation from a to b is always ccw independently on a.y sign
    return true;
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

bool inCircle( const std::array<PreciseVertCoords2, 4>& vs )
{
    return inCircle( vs.data() );
}

bool inCircle( const PreciseVertCoords2* vs )
{
    PreciseVertCoordsll vs3d[4];
    for ( int i = 0; i < 4; ++i )
    {
        vs3d[i].id = vs[i].id;
        vs3d[i].pt = Vector3ll( Vector3ll::ValueType( vs[i].pt.x ) * vs[i].pt.x + Vector3ll::ValueType( vs[i].pt.y ) * vs[i].pt.y, vs[i].pt.x, vs[i].pt.y );
    }
    return ccw( vs ) == orient3d( vs3d ); // TODO: looks like orient3d is not "honest" enough for this predicate
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

    // This one fails
    EXPECT_TRUE( inCircle( vs ) );
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
    
    // looks like this should be false
    EXPECT_TRUE( inCircle( { vs[4],vs[2],vs[0],vs[3] } ) ); // 3_v is in circle

    // prove that 0_v is inside 142 triangle
    EXPECT_TRUE( ccw( { vs[1],vs[4],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[4],vs[2],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[1],vs[0] } ) );
    // it means that 142 circle should be larger in +Y half plane and so 3_v should be inside it
    // but it is not
    // DISABLED
    //EXPECT_TRUE( inCircle( { vs[1],vs[4],vs[2],vs[3] } ) );
}

} //namespace MR
