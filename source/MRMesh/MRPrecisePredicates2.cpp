#include "MRPrecisePredicates2.h"
#include "MRHighPrecision.h"
#include "MRPrecisePredicates3.h"
#include "MRSparsePolynomial.h"
#include "MRDivRound.h"

namespace MR
{

namespace
{

struct PointDegree
{
    Vector2i pt;
    int d = 0; // degree of epsilon for pt.y
};

template<size_t N>
std::array<PointDegree, N> getPointDegrees( const std::array<PreciseVertCoords2, N> & vs )
{
    struct VertN
    {
        VertId v;
        int n = 0;
    };
    std::array<VertN, N> as;
    for ( int i = 0; i < N; ++i )
        as[i] = { vs[i].id, i };
    std::sort( begin( as ), end( as ), []( const auto & a, const auto & b ) { return a.v < b.v; } );

    std::array<PointDegree, N> res;
    int d = 1;
    for ( int i = 0; i < N; ++i )
    {
        assert( i == 0 || as[i-1].v < as[i].v ); // no duplicate vertices are permitted
        const auto n = as[i].n;
        res[n] = { vs[n].pt, d };
        d *= 9;
    }
    return res;
}

// Int64 is enough to store all coefficients in ( ccw(sa,s[0])*ccw(sb,s[1])   -   ccw(sb,s[0])*ccw(sa,s[1]) ) except for degree 0, which is computed separately.
template<int M>
using Poly = SparsePolynomial<Int64, int, M>;

template<int M>
Poly<M> ccwPoly( const PointDegree & a, const PointDegree & b, const PointDegree & c,
    int db ) // degree.x = degree.y * db
{
    const Poly<M> xx( a.pt.x - c.pt.x, a.d * db, 1, c.d * db, -1 );
    const Poly<M> xy( a.pt.y - c.pt.y, a.d     , 1, c.d     , -1 );
    const Poly<M> yx( b.pt.x - c.pt.x, b.d * db, 1, c.d * db, -1 );
    const Poly<M> yy( b.pt.y - c.pt.y, b.d     , 1, c.d     , -1 );
    auto det = xx * yy;
    det -= xy * yx;
    return det;
}

Int64 area( const Vector2i & a, const Vector2i & b, const Vector2i & c )
{
    const Int64 xx( a.x - c.x );
    const Int64 xy( a.y - c.y );
    const Int64 yx( b.x - c.x );
    const Int64 yy( b.y - c.y );
    return xx * yy - xy * yx;
}

} // anonymous namespace

// see https://arxiv.org/pdf/math/9410209 Table 4-i:
// a=(pi_i,1, pi_i,2)
// b=(pi_j,1, pi_j,2)
bool ccw( const Vector2i & a, const Vector2i & b )
{
    if ( auto v = cross( Vector2i64{ a }, Vector2i64{ b } ) )
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

bool smaller2( const std::array<PreciseVertCoords2, 4> & vs )
{
    if ( auto d = area( vs[0].pt, vs[1].pt, vs[2].pt ) - area( vs[0].pt, vs[1].pt, vs[3].pt ) )
        return d < 0;

    // areas are equal, apply simulation-of-simplicity
    const auto ds = getPointDegrees( vs );

    // 84 was found experimentally to be enough for all cases with 4 points having equal coordinates (but different ids);
    // if it is not enough then we will get assert violation inside poly.isPositive(), and increase the value
    constexpr int MaxD = 84;
    auto poly = ccwPoly<MaxD>( ds[0], ds[1], ds[2], 3 );
        poly -= ccwPoly<MaxD>( ds[0], ds[1], ds[3], 3 );

    return !poly.isPositive();
}

bool orientParaboloid3d( const Vector2i & a0, const Vector2i & b0, const Vector2i & c0 )
{
    const Vector3i64 a( a0.x, a0.y, sqr( Int64( a0.x ) ) + sqr( Int64( a0.y ) ) );
    const Vector3i64 b( b0.x, b0.y, sqr( Int64( b0.x ) ) + sqr( Int64( b0.y ) ) );
    const Vector3i64 c( c0.x, c0.y, sqr( Int64( c0.x ) ) + sqr( Int64( c0.y ) ) );

    //e**0
    if ( auto v = mixed( Vector3i128fast( a ), Vector3i128fast( b ), Vector3i128fast( c ) ) )
        return v > 0;

    // e**1
    const auto bxy_cxy = cross( Vector2i64{ b.x, b.y }, Vector2i64{ c.x, c.y } );
    if ( auto v = -cross( Vector2i128fast{ b.x, b.z }, Vector2i128fast{ c.x, c.z } ) + 2 * a.y * FastInt128( bxy_cxy ) )
        return v > 0;

    // e**2
    if ( auto v = bxy_cxy )
        return v > 0;

    // e**3
    assert( bxy_cxy == 0 );
    if ( auto v = cross( Vector2i128fast{ b.y, b.z }, Vector2i128fast{ c.y, c.z } ) ) // + 2 * a.x * bxy_cxy;
        return v > 0;

    // e**6 same as e**2

    // e**9
    const auto axy_cxy = cross( Vector2i64{ a.x, a.y }, Vector2i64{ c.x, c.y } );
    if ( auto v = cross( Vector2i128fast{ a.x, a.z }, Vector2i128fast{ c.x, c.z } ) - 2 * b.y * FastInt128( axy_cxy ) )
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
    if ( auto v = b.x * FastInt128( a.z ) - a.x * FastInt128( b.z ) )
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

bool segmentIntersectionOrder( const std::array<PreciseVertCoords2, 6> & vs )
{
    // s=01, sa=23, sb=45
    assert( doSegmentSegmentIntersect( { vs[0], vs[1], vs[2], vs[3] } ) );
    assert( doSegmentSegmentIntersect( { vs[0], vs[1], vs[4], vs[5] } ) );

    // if sa and sb have a shared point
    PreciseVertCoords2 sharedPoint;
    for ( auto va : { vs[2], vs[3] } )
        for ( auto vb : { vs[4], vs[5] } )
            if ( va.id == vb.id )
            {
                assert( va.pt == vb.pt );
                sharedPoint = va;
                goto exitLoop;
            }
    exitLoop:

    if ( sharedPoint.id )
    {
        // segments sa and sb have one shared point
        auto secondPointB = ( sharedPoint.id == vs[4].id ) ? vs[5] : vs[4];
        return ccw( { vs[2], vs[3], secondPointB } ) == ccw( { vs[2], vs[3], vs[1] } );
    }
    else
    {
        // segments sa and sb have no shared points
        const bool a1 = ccw( { vs[4], vs[5], vs[2] } );
        if ( a1 == ccw( { vs[4], vs[5], vs[3] } ) )
        {
            // all a-points are on one side of sb
            return a1 == ccw( { vs[4], vs[5], vs[0] } );
        }

        const bool b1 = ccw( { vs[2], vs[3], vs[4] } );
        if ( b1 == ccw( { vs[2], vs[3], vs[5] } ) )
        {
            // all b-points are on one side of sa
            return b1 == ccw( { vs[2], vs[3], vs[1] } );
        }

        // segments sa and sb intersect one another, process it as general case
    }

    // res = ( ccw(sa,s[0])*ccw(sb,s[1])   -   ccw(sb,s[0])*ccw(sa,s[1]) ) /
    //       ( ccw(sa,s[0])-ccw(sa,s[1]) ) * ( ccw(sb,s[0])-ccw(sb,s[1]) )
    const auto areaSaOrg  = area( vs[2].pt, vs[3].pt, vs[0].pt );
    const auto areaSaDest = area( vs[2].pt, vs[3].pt, vs[1].pt );
    assert( ( areaSaOrg <= 0 && areaSaDest >= 0 ) || ( areaSaOrg >= 0 && areaSaDest <= 0 ) );

    const auto areaSbOrg  = area( vs[4].pt, vs[5].pt, vs[0].pt );
    const auto areaSbDest = area( vs[4].pt, vs[5].pt, vs[1].pt );
    assert( ( areaSbOrg <= 0 && areaSbDest >= 0 ) || ( areaSbOrg >= 0 && areaSbDest <= 0 ) );

    const auto nomSimple = FastInt128( areaSaOrg ) * FastInt128( areaSbDest ) - FastInt128( areaSbOrg ) * FastInt128( areaSaDest );
    if ( nomSimple != 0 )
    {
        // happy not-degenerated path
        bool res = nomSimple > 0;
        assert( areaSaOrg || areaSaDest );
        if ( areaSaOrg < areaSaDest )
            res = !res;
        assert( areaSbOrg || areaSbDest );
        if ( areaSbOrg < areaSbDest )
            res = !res;
        return res;
    }

    const auto ds = getPointDegrees( vs );

    // 840 was found experimentally to be enough for all cases with 6 points having equal coordinates (but different ids);
    // if it is not enough then we will get assert violation inside poly.isPositive(), and increase the value
    constexpr int MaxD = 840;
    const auto polySaOrg  = ccwPoly<MaxD>( ds[2], ds[3], ds[0], 3 );
    const auto polySaDest = ccwPoly<MaxD>( ds[2], ds[3], ds[1], 3 );
    assert( !polySaOrg.empty() || !polySaDest.empty() );
    assert( polySaOrg.empty() || polySaDest.empty() || polySaOrg.isPositive() != polySaDest.isPositive() );
    const bool posSaOrg = polySaOrg.empty() ? !polySaDest.isPositive() : polySaOrg.isPositive();

    const auto polySbOrg  = ccwPoly<MaxD>( ds[4], ds[5], ds[0], 3 );
    const auto polySbDest = ccwPoly<MaxD>( ds[4], ds[5], ds[1], 3 );
    assert( !polySbOrg.empty() || !polySbDest.empty() );
    assert( polySbOrg.empty() || polySbDest.empty() || polySbOrg.isPositive() != polySbDest.isPositive() );
    const bool posSbOrg = polySbOrg.empty() ? !polySbDest.isPositive() : polySbOrg.isPositive();

    auto nom = polySaOrg * polySbDest;
    nom -= polySbOrg * polySaDest;

    // nomSimple == 0 means that zero degree coefficient is zero, but it can be computed incorrectly due overflow errors in 64-bit arithmetic
    nom.setZeroCoeff( 0 );

    bool res = nom.isPositive();
    if ( posSaOrg != posSbOrg ) // denominator is negative
        res = !res;
    return res;
}

Vector2i findSegmentSegmentIntersectionPrecise(
    const Vector2i& ai, const Vector2i& bi, const Vector2i& ci, const Vector2i& di )
{
    auto abc = cross( Vector2i64( ai - ci ), Vector2i64( bi - ci ) );
    if ( abc < 0 )
        abc = -abc;
    auto abd = cross( Vector2i64( ai - di ), Vector2i64( bi - di ) );
    if ( abd < 0 )
        abd = -abd;
    const auto sum = abc + abd;
    if ( sum != 0 )
        return Vector2i( divRound( FastInt128( abc ) * Vector2i128fast( di ) + FastInt128( abd ) * Vector2i128fast( ci ), FastInt128( sum ) ) );

    // degenerate case
    auto adLSq = Vector2i64( di - ai ).lengthSq();
    auto bcLSq = Vector2i64( bi - ci ).lengthSq();
    if ( adLSq > bcLSq )
        return ci;
    else if ( bcLSq > adLSq )
        return di;
    else
        return Vector2i( divRound( Vector2i64( ai ) + Vector2i64( bi ) + Vector2i64( ci ) + Vector2i64( di ), Int64( 2 ) ) );
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

} //namespace MR
