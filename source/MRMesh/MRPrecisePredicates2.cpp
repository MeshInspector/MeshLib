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

bool orientParaboloid3d( const Vector2i & a0, const Vector2i & b0, const Vector2i & c0 )
{
    Vector3ll a( a0.x, a0.y, sqr( (long long) a0.x ) + sqr( (long long) a0.y ) );
    Vector3ll b( b0.x, b0.y, sqr( (long long) b0.x ) + sqr( (long long) b0.y ) );
    Vector3ll c( c0.x, c0.y, sqr( (long long) c0.x ) + sqr( (long long) c0.y ) );

    using Vector2hp = Vector2<HighPrecisionInt>;
    using Vector3hpp = Vector3<HighHighPrecisionInt>;

    //e**0
    if ( auto v = mixed( Vector3hpp{ a }, Vector3hpp{ b }, Vector3hpp{ c } ) )
        return v > 0;

    // e**1
    const auto bxy_cxy = cross( Vector2hp{ b.x, b.y }, Vector2hp{ c.x, c.y } );
    if ( auto v = -cross( Vector2hp{ b.x, b.z }, Vector2hp{ c.x, c.z } ) + 2 * a.y * bxy_cxy )
        return v > 0;

    // e**2
    if ( auto v = bxy_cxy )
        return v > 0;

    // e**3
    assert( bxy_cxy == 0 );
    if ( auto v = cross( Vector2hp{ b.y, b.z }, Vector2hp{ c.y, c.z } ) ) // + 2 * a.x * bxy_cxy;
        return v > 0;

    // e**6 same as e**2

    // e**9
    const auto axy_cxy = cross( Vector2hp{ a.x, a.y }, Vector2hp{ c.x, c.y } );
    if ( auto v = cross( Vector2hp{ a.x, a.z }, Vector2hp{ c.x, c.z } ) - 2 * b.y * axy_cxy )
        return v > 0;

    // e**10
    if ( auto v = c.x * ( b.y - a.y ) )
        return v > 0;

    // e**11
    if ( auto v = -c.x )
        return v > 0;

    // e**12: -2*a.x*c.x - 2*b.y*c.y + c.z
    assert( c.x == 0 );
    if ( auto v = - 2 * b.y * c.y + c.z )
        return v > 0;

    // e**18
    if ( auto v = -axy_cxy )
        return v > 0;

    // e**21
    if ( auto v = -c.y )
        return v > 0;
    assert( c.x == 0 && c.y == 0 && c.z == 0 );

    // e**81
    if ( auto v = b.x * HighPrecisionInt( a.z ) - a.x * HighPrecisionInt( b.z ) )
        return v > 0;

    // e**82
    if ( auto v = a.y * b.x )
        return v > 0;

    // e**83
    if ( auto v = b.x )
        return v > 0;

    // e**84
    if ( auto v = -b.z )
        return v > 0; // can only be false, since b.z >= 0
    assert( b.x == 0 && b.y == 0 && b.z == 0 );

    // e**99
    if ( auto v = -a.x )
        return v > 0;

    // e**102
    return false;
}

bool orientParaboloid3d( const PreciseVertCoords2* vs )
{
    bool odd = false;
    std::array<int, 4> order = { 0, 1, 2, 3 };

    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = i + 1; j < 4; ++j )
        {
            assert( vs[order[i]].id != vs[order[j]].id );
            if ( vs[order[i]].id > vs[order[j]].id )
            {
                odd = !odd;
                std::swap( order[i], order[j] );
            }
        }
    }

    return odd != orientParaboloid3d( vs[order[0]].pt, vs[order[1]].pt, vs[order[2]].pt, vs[order[3]].pt );
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
    // orientParaboloid3d and not ordinary orient3d as in SoS article, since additional coordinate x*x+y*y is not independent from x and y
    return ccw( vs ) == orientParaboloid3d( vs );
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

    // Check that vs[3] is inCircle
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
    
    EXPECT_FALSE( inCircle( { vs[4],vs[2],vs[0],vs[3] } ) ); // 3_v is in circle

    // prove that 0_v is inside 142 triangle
    EXPECT_TRUE( ccw( { vs[1],vs[4],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[4],vs[2],vs[0] } ) );
    EXPECT_TRUE( ccw( { vs[2],vs[1],vs[0] } ) );
    // it means that 142 circle should be larger in +Y half plane and so 3_v should be inside it
    EXPECT_FALSE( inCircle( { vs[1],vs[4],vs[2],vs[3] } ) );
}

} //namespace MR
