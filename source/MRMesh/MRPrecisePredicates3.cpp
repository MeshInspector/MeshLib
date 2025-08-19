#include "MRPrecisePredicates3.h"
#include "MRHighPrecision.h"
#include "MRVector2.h"
#include "MRBox.h"
#include "MRDivRound.h"
#include "MRSparsePolynomial.h"
#include <optional>

namespace MR
{

namespace
{
// INT_MAX in double for mapping in int range
constexpr double cRangeIntMax = 0.99 * std::numeric_limits<int>::max(); // 0.99 to be sure the no overflow will ever happen due to rounding errors

struct PointDegree
{
    Vector3i pt;
    int d = 0; // degree of epsilon for pt.z
};

// this value was found experimentally for segmentIntersectionOrder with all 8 points have equal coordinates (but different ids),
// if it is not enough then we will get assert violation inside poly.isPositive(), and increase the value
constexpr int cMaxPolyD = 14'941'836;

std::array<PointDegree, 8> getPointDegrees( const std::array<PreciseVertCoords, 8> & vs )
{
    struct VertN
    {
        VertId v;
        int n = 0;
    };
    std::array<VertN, 8> as;
    for ( int i = 0; i < 8; ++i )
        as[i] = { vs[i].id, i };
    std::sort( begin( as ), end( as ), []( const auto & a, const auto & b ) { return a.v < b.v; } );

    std::array<PointDegree, 8> res;
    int d = 1;
    constexpr int maxD = INT_MAX / 9;
    static_assert( maxD > cMaxPolyD );
    constexpr int preMaxD = maxD / 27;
    for ( int i = 0; i < 8; ++i )
    {
        const auto n = as[i].n;
        res[n] = { vs[n].pt, d };
        if ( i < 7 && as[i].v < as[i+1].v ) // skip to support triangles with shared vertices
        {
            if ( d <= preMaxD )
                d *= 27; // normal power up
            else if ( d <= maxD )
                d = maxD; // to avoid integer overflow in orient3dPoly, assuming that such huge powers will never be necessary
        }
    }
    return res;
}

// 128 bits are enough to store all coefficients in ( orient3d(ta,s[0])*orient3d(tb,s[1]) - orient3d(tb,s[0])*orient3d(ta,s[1]) )
// except for degree 0, which is computed separately.
using Poly = SparsePolynomial<FastInt128, int, cMaxPolyD>;

Poly orient3dPoly( const PointDegree & a, const PointDegree & b, const PointDegree & c, const PointDegree & d,
    int dy ) // degree.x = ( degree.y = degree.z * dy ) * dy
{
    const int dx = dy * dy;

    const Poly xx( a.pt.x - d.pt.x, a.d * dx, 1, d.d * dx, -1 );
    const Poly xy( a.pt.y - d.pt.y, a.d * dy, 1, d.d * dy, -1 );
    const Poly xz( a.pt.z - d.pt.z, a.d     , 1, d.d     , -1 );

    const Poly yx( b.pt.x - d.pt.x, b.d * dx, 1, d.d * dx, -1 );
    const Poly yy( b.pt.y - d.pt.y, b.d * dy, 1, d.d * dy, -1 );
    const Poly yz( b.pt.z - d.pt.z, b.d     , 1, d.d     , -1 );

    const Poly zx( c.pt.x - d.pt.x, c.d * dx, 1, d.d * dx, -1 );
    const Poly zy( c.pt.y - d.pt.y, c.d * dy, 1, d.d * dy, -1 );
    const Poly zz( c.pt.z - d.pt.z, c.d     , 1, d.d     , -1 );

    Poly t;

    t  = yy * zz;
    t -= yz * zy;
    Poly det = xx * t;

    t  = yx * zz;
    t -= yz * zx;
    det -= xy * t;

    t  = yx * zy;
    t -= yy * zx;
    det += xz * t;

    return det;
}

Int128 volume( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
{
    const Vector3i64 x( a - d );
    const Vector3i64 y( b - d );
    const Vector3i64 z( c - d );

    return
        x.x * Int128( y.y * z.z - y.z * z.y )
     -  x.y * Int128( y.x * z.z - y.z * z.x )
     +  x.z * Int128( y.x * z.y - y.y * z.x );
}

} // anonymous namespace

bool orient3d( const Vector3i & a, const Vector3i& b, const Vector3i& c )
{
    auto vhp = dot( Vector3i128fast{ a }, Vector3i128fast{ cross( Vector3i64{ b }, Vector3i64{ c } ) } );
    if ( vhp ) return vhp > 0;

    auto v = cross( Vector2i64{ b.x, b.y }, Vector2i64{ c.x, c.y } );
    if ( v ) return v > 0;

    v = -cross( Vector2i64{ b.x, b.z }, Vector2i64{ c.x, c.z } );
    if ( v ) return v > 0;

    v = cross( Vector2i64{ b.y, b.z }, Vector2i64{ c.y, c.z } );
    if ( v ) return v > 0;

    v = -cross( Vector2i64{ a.x, a.y }, Vector2i64{ c.x, c.y } );
    if ( v ) return v > 0;

    if ( c.x ) return c.x > 0;

    if ( c.y ) return c.y < 0;

    v = cross( Vector2i64{ a.x, a.z }, Vector2i64{ c.x, c.z } );
    if ( v ) return v > 0;

    if ( c.z ) return c.z > 0;

#ifndef NDEBUG
    v = -cross( Vector2i64{ a.y, a.z }, Vector2i64{ c.y, c.z } );
    assert( v == 0 );
    if ( v ) return v > 0;
#endif

    v = cross( Vector2i64{ a.x, a.y }, Vector2i64{ b.x, b.y } );
    if ( v ) return v > 0;

    if ( b.x ) return b.x < 0;

    if ( b.y ) return b.y > 0;

    if ( a.x ) return a.x > 0;

    return true;
}

bool orient3d( const PreciseVertCoords* vs )
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

    return odd != orient3d( vs[order[0]].pt, vs[order[1]].pt, vs[order[2]].pt, vs[order[3]].pt );
}

bool orient3d( const std::array<PreciseVertCoords, 4> & vs )
{
    return orient3d( vs.data() );
}

TriangleSegmentIntersectResult doTriangleSegmentIntersect( const std::array<PreciseVertCoords, 5> & vs )
{
    TriangleSegmentIntersectResult res;
    constexpr int a = 0;
    constexpr int b = 1;
    constexpr int c = 2;
    constexpr int d = 3;
    constexpr int e = 4;

    auto orient3d = [&]( int p, int q, int r, int s )
    {
        return MR::orient3d( { vs[p], vs[q], vs[r], vs[s] } );
    };

    const auto abcd = orient3d( a, b, c, d );
    res.dIsLeftFromABC = abcd;
    const auto abce = orient3d( a, b, c, e );
    if ( abcd == abce )
        return res; // segment DE is located at one side of the plane ABC

    const auto dabe = orient3d( a, b, d, e );
    const auto dbce = orient3d( b, c, d, e );
    if ( dabe != dbce )
        return res; // segment AC is located at one side of the plane DEB

    const auto dcae = !orient3d( a, c, d, e ); // '!' is due to inverted order of a and c
    if ( dbce != dcae )
        return res; // segment AB is located at one side of the plane DEC

    assert ( dcae == dabe ); // segment BC is crossed by the plane DEA

    res.doIntersect = true;
    return res;
}

bool segmentIntersectionOrder( const std::array<PreciseVertCoords, 8> & vs )
{
    // s=01, ta=234, tb=567
    auto as = { vs[2], vs[3], vs[4] };
    auto bs = { vs[5], vs[6], vs[7] };

    assert( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) );
    assert( doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) );

    // check for shared points in ta and tb
    PreciseVertCoords firstSharedPoint;
    for ( auto va : as )
        for ( auto vb : bs )
            if ( va.id == vb.id )
            {
                assert( va.pt == vb.pt );
                firstSharedPoint = va;
                goto exitLoop1;
            }
    exitLoop1:

    if ( firstSharedPoint.id )
    {
        PreciseVertCoords secondSharedPoint;
        for ( auto va : as )
            for ( auto vb : bs )
                if ( va.id == vb.id && va.id != firstSharedPoint.id )
                {
                    assert( va.pt == vb.pt );
                    secondSharedPoint = va;
                    goto exitLoop2;
                }
        exitLoop2:

        if ( secondSharedPoint.id )
        {
            PreciseVertCoords thirdPointB;
            for ( auto vb : bs )
                if ( vb.id != firstSharedPoint.id && vb.id != secondSharedPoint.id )
                {
                    thirdPointB = vb;
                    break;
                }
            assert( thirdPointB.id );
            return orient3d( { vs[2], vs[3], vs[4], thirdPointB } )
                == orient3d( { vs[2], vs[3], vs[4], vs[1] } );
        }

        // only one shared point in ta and tb

        PreciseVertCoords secondPointA, thirdPointA;
        for ( auto va : as )
            if ( va.id != firstSharedPoint.id )
            {
                if ( !secondPointA.id )
                    secondPointA = va;
                else
                    thirdPointA = va;
            }
        assert( secondPointA.id && thirdPointA.id );
        const bool a2 = orient3d( { vs[5], vs[6], vs[7], secondPointA } );
        if ( a2 == orient3d( { vs[5], vs[6], vs[7], thirdPointA } ) ) //both not-shared a-points are on one side of tb
            return a2 == orient3d( { vs[5], vs[6], vs[7], vs[0] } );

        PreciseVertCoords secondPointB, thirdPointB;
        for ( auto vb : bs )
            if ( vb.id != firstSharedPoint.id )
            {
                if ( !secondPointB.id )
                    secondPointB = vb;
                else
                    thirdPointB = vb;
            }
        assert( secondPointB.id && thirdPointB.id );
        const bool b2 = orient3d( { vs[2], vs[3], vs[4], secondPointB } );
        if ( b2 == orient3d( { vs[2], vs[3], vs[4], thirdPointB } ) ) //both not-shared b-points are on one side of ta
            return b2 == orient3d( { vs[2], vs[3], vs[4], vs[1] } );

        // triangles ta and tb intersect one another, process it as general case
    }
    else
    {
        // no shared points in ta and tb
        const bool a1 = orient3d( { vs[5], vs[6], vs[7], vs[2] } );
        if ( a1 == orient3d( { vs[5], vs[6], vs[7], vs[3] } ) && a1 == orient3d( { vs[5], vs[6], vs[7], vs[4] } ) )
        {
            // all a-points are on one side of tb
            return a1 == orient3d( { vs[5], vs[6], vs[7], vs[0] } );
        }

        const bool b1 = orient3d( { vs[2], vs[3], vs[4], vs[5] } );
        if ( b1 == orient3d( { vs[2], vs[3], vs[4], vs[6] } ) && b1 == orient3d( { vs[2], vs[3], vs[4], vs[7] } ) )
        {
            // all b-points are on one side of ta
            return b1 == orient3d( { vs[2], vs[3], vs[4], vs[1] } );
        }

        // triangles ta and tb intersect one another, process it as general case
    }

    // res = ( orient3d(ta,s[0])*orient3d(tb,s[1])   -   orient3d(tb,s[0])*orient3d(ta,s[1]) ) /
    //       ( orient3d(ta,s[0])-orient3d(ta,s[1]) ) * ( orient3d(tb,s[0])-orient3d(tb,s[1]) )
    const auto volumeTaOrg  = volume( vs[2].pt, vs[3].pt, vs[4].pt, vs[0].pt );
    const auto volumeTaDest = volume( vs[2].pt, vs[3].pt, vs[4].pt, vs[1].pt );
    assert( ( volumeTaOrg <= 0 && volumeTaDest >= 0 ) || ( volumeTaOrg >= 0 && volumeTaDest <= 0 ) );

    const auto volumeTbOrg  = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[0].pt );
    const auto volumeTbDest = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[1].pt );
    assert( ( volumeTbOrg <= 0 && volumeTbDest >= 0 ) || ( volumeTbOrg >= 0 && volumeTbDest <= 0 ) );

    const auto nomSimple = Int256( volumeTaOrg ) * Int256( volumeTbDest ) - Int256( volumeTbOrg ) * Int256( volumeTaDest );
    if ( nomSimple != 0 )
    {
        // happy not-degenerated path
        bool res = nomSimple > 0;
        assert( volumeTaOrg || volumeTaDest );
        if ( volumeTaOrg < volumeTaDest )
            res = !res;
        assert( volumeTbOrg || volumeTbDest );
        if ( volumeTbOrg < volumeTbDest )
            res = !res;
        return res;
    }

    const auto ds = getPointDegrees( vs );

    const auto polyTaOrg  = orient3dPoly( ds[2], ds[3], ds[4], ds[0], 3 );
    const auto polyTaDest = orient3dPoly( ds[2], ds[3], ds[4], ds[1], 3 );
    assert( !polyTaOrg.empty() || !polyTaDest.empty() );
    assert( polyTaOrg.empty() || polyTaDest.empty() || polyTaOrg.isPositive() != polyTaDest.isPositive() );
    const bool posTaOrg = polyTaOrg.empty() ? !polyTaDest.isPositive() : polyTaOrg.isPositive();

    const auto polyTbOrg  = orient3dPoly( ds[5], ds[6], ds[7], ds[0], 3 );
    const auto polyTbDest = orient3dPoly( ds[5], ds[6], ds[7], ds[1], 3 );
    assert( !polyTbOrg.empty() || !polyTbDest.empty() );
    assert( polyTbOrg.empty() || polyTbDest.empty() || polyTbOrg.isPositive() != polyTbDest.isPositive() );
    const bool posTbOrg = polyTbOrg.empty() ? !polyTbDest.isPositive() : polyTbOrg.isPositive();

    auto nom = polyTaOrg * polyTbDest;
    nom -= polyTbOrg * polyTaDest;

    // nomSimple == 0 means that zero degree coefficient is zero, but it can be computed incorrectly due overflow errors in 128-bit arithmetic
    nom.setZeroCoeff( 0 );

    bool res = nom.isPositive();
    if ( posTaOrg != posTbOrg ) // denominator is negative
        res = !res;
    return res;
}

ConvertToIntVector getToIntConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double invRange = cRangeIntMax / maxDim;

    return [invRange, center] ( const Vector3f& v )
    {
        // perform intermediate operations in double for better precision
        const auto d = ( Vector3d{ v } - center ) * invRange;
        // and round to the nearest integer instead of truncating to zero
        return Vector3i( (int)std::round( d.x ), (int)std::round( d.y ), (int)std::round( d.z ) );
    };
}

ConvertToFloatVector getToFloatConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double range = maxDim / cRangeIntMax;

    return [range, center] ( const Vector3i& v )
    {
        return Vector3f( Vector3d{ v }*range + center );
    };
}

std::optional<Vector3i> findTwoSegmentsIntersection( const Vector3i& ai, const Vector3i& bi, const Vector3i& ci, const Vector3i& di )
{
    const auto ab = Vector3i64{ bi - ai };
    const auto ac = Vector3i64{ ci - ai };
    const auto ad = Vector3i64{ di - ai };
    const auto abc = cross( ab, ac );
    const auto abd = cross( ab, ad );

    if ( dot( Vector3i128fast( abc ), Vector3i128fast( abd ) ) > 0 )
        return std::nullopt; // CD is on one side of AB

    const auto cd = Vector3i64{ di - ci };
    const auto cb = Vector3i64{ bi - ci };
    const auto cda = cross( cd, -ac );
    const auto cdb = cross( cd, cb );
    if ( dot( Vector3i128fast( cda ), Vector3i128fast( cdb ) ) > 0 )
        return std::nullopt; // AB is on one side of CD

    constexpr Vector3i64 zero;
    if ( ( abc == zero && abd == zero ) || ( cda == zero && cdb == zero ) ) // collinear
    {
        const auto dAC = dot( ab, ac );
        const auto dAD = dot( ab, ad );
        if ( dAC < 0 && dAD < 0 )
            return std::nullopt; // both C and D are lower than A (on the AB segment)

        const auto dBC = dot( -ab, -cb );
        const auto dBD = dot( -ab, Vector3i64{ di - bi } );
        if ( dBC < 0 && dBD < 0 )
            return std::nullopt; // both C and D are greater than B (on the AB segment)

        // have common points
        auto onePoint = dAC < 0 ? ai : ci; // find point that is closer to B
        auto otherPoint = dBD < 0 ? bi : di; // find point that is closer to A
        return ( onePoint + otherPoint ) / 2; // return middle point of overlapping segment
    }

    // common intersection - AB and CD are non-collinear
    const Vector3i64 n = abc - abd; // not unit
    FastInt128 ck = dot( Vector3i128fast( n ), Vector3i128fast( abc ) );
    FastInt128 dk = dot( Vector3i128fast( n ), Vector3i128fast( abd ) );
    assert( ck >=0 && dk <= 0 );

    // scale down ck and dk to make sure that below products can be computed in 128 bits
    // assume that abs( di ) <= 2^30 and abs( ci ) <= 2^30
    constexpr FastInt128 x = FastInt128( 1 ) << 96; //2^96
    if ( ck > x || -dk > x )
    {
        ck = ck >> 32;
        dk = -( (-dk) >> 32 );
    }
    return Vector3i( divRound( ck * Vector3i128fast{ di } - dk * Vector3i128fast{ ci }, ck - dk ) );
}

Vector3f findTriangleSegmentIntersectionPrecise(
    const Vector3f& a, const Vector3f& b, const Vector3f& c,
    const Vector3f& d, const Vector3f& e, 
    CoordinateConverters converters )
{
    auto ai = converters.toInt( a );
    auto bi = converters.toInt( b );
    auto ci = converters.toInt( c );
    auto di = converters.toInt( d );
    auto ei = converters.toInt( e );
    auto abcd = dot( Vector3i128fast{ ai - di }, Vector3i128fast{ cross( Vector3i64{ bi - di }, Vector3i64{ ci - di } ) } );
    if ( abcd < 0 )
        abcd = -abcd;
    auto abce = dot( Vector3i128fast{ ai - ei }, Vector3i128fast{ cross( Vector3i64{ bi - ei }, Vector3i64{ ci - ei } ) } );
    if ( abce < 0 )
        abce = -abce;
    auto sum = abcd + abce;
    if ( sum != 0 )
        return converters.toFloat( Vector3i{ divRound( abcd * Vector3i128fast{ ei } + abce * Vector3i128fast{ di }, sum ) } );
    // rare case when `sum == 0` 
    // suggest finding middle point of edge segment laying inside triangle
    Vector3i64 sumVec;
    int numSum = 0;
    if ( auto iABDE = findTwoSegmentsIntersection( ai, bi, di, ei ) )
    {
        sumVec += Vector3i64{ *iABDE };
        ++numSum;
    }
    if ( auto iBCDE = findTwoSegmentsIntersection( bi, ci, di, ei ) )
    {
        sumVec += Vector3i64{ *iBCDE };
        ++numSum;
    }
    if ( auto iCADE = findTwoSegmentsIntersection( ci, ai, di, ei ) )
    {
        sumVec += Vector3i64{ *iCADE };
        ++numSum;
    }
    if ( numSum > 0 )
        return converters.toFloat( Vector3i{ Vector3d( sumVec ) / double( numSum ) } );

    // rare case when `numSum == 0` - segment is fully inside face
    return Vector3f( ( Vector3d( d ) + Vector3d( e ) ) * 0.5 );
}

} //namespace MR
