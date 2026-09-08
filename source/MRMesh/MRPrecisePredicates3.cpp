#include "MRPrecisePredicates3.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRVector.h"
#include "MRFastInt.h"
#include "MRHighPrecision.h"
#include "MRInt64Mul128.h"
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
    std::int64_t d = 0; // degree of epsilon for pt.z; pt.y gets 3*d, pt.x gets 9*d
};

// these values were found experimentally as the largest degree of a polynomial term that must be stored in the predicates
// (the leading term of nom, or the leading term of at least one of two orient3d-polynomials for the segment's ends)
// in the tests and in a random sweep of degenerate inputs; if it is not enough then we will get assert violation inside
// poly.isPositive(), and increase the value; all polynomial terms of higher degrees are not stored to save computation time
constexpr std::int64_t cMaxPolyDTriTri   = 15'943'959; // segmentIntersectionOrder
constexpr std::int64_t cMaxPolyDTriPlane = 430'486'893; // segmentIntersectionTriPlaneOrder: it reaches the polynomial path with all three largest ids in the plane

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
    std::int64_t d = 1;
    for ( int i = 0; i < 8; ++i )
    {
        const auto n = as[i].n;
        res[n] = { vs[n].pt, d };
        if ( i < 7 && as[i].v < as[i+1].v ) // skip to support triangles with shared vertices
            d *= 27;
    }
    return res;
}

// 128 bits are enough to store all coefficients of one orient3d-polynomial (products of three coordinate differences),
// while the coefficients of the products of two such polynomials need up to 256 bits, see mulAs<Int128Mul256> below
template <std::int64_t M>
using Poly = SparsePolynomial<FastInt128, std::int64_t, M>;

template <std::int64_t M>
Poly<M> orient3dPoly( const PointDegree & a, const PointDegree & b, const PointDegree & c, const PointDegree & d,
    std::int64_t dy ) // degree.x = ( degree.y = degree.z * dy ) * dy
{
    const std::int64_t dx = dy * dy;

    const Poly<M> xx( a.pt.x - d.pt.x, a.d * dx, 1, d.d * dx, -1 );
    const Poly<M> xy( a.pt.y - d.pt.y, a.d * dy, 1, d.d * dy, -1 );
    const Poly<M> xz( a.pt.z - d.pt.z, a.d     , 1, d.d     , -1 );

    const Poly<M> yx( b.pt.x - d.pt.x, b.d * dx, 1, d.d * dx, -1 );
    const Poly<M> yy( b.pt.y - d.pt.y, b.d * dy, 1, d.d * dy, -1 );
    const Poly<M> yz( b.pt.z - d.pt.z, b.d     , 1, d.d     , -1 );

    const Poly<M> zx( c.pt.x - d.pt.x, c.d * dx, 1, d.d * dx, -1 );
    const Poly<M> zy( c.pt.y - d.pt.y, c.d * dy, 1, d.d * dy, -1 );
    const Poly<M> zz( c.pt.z - d.pt.z, c.d     , 1, d.d     , -1 );

    Poly<M> t;

    t  = yy * zz;
    t -= yz * zy;
    Poly<M> det = xx * t;

    t  = yx * zz;
    t -= yz * zx;
    det -= xy * t;

    t  = yx * zy;
    t -= yy * zx;
    det += xz * t;

    return det;
}

FastInt128 volume( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
{
    const Vector3i64 x( a - d );
    const Vector3i64 y( b - d );
    const Vector3i64 z( c - d );

    return
        Int64Mul128( x.x ) * Int64Mul128( y.y * z.z - y.z * z.y )
     -  Int64Mul128( x.y ) * Int64Mul128( y.x * z.z - y.z * z.x )
     +  Int64Mul128( x.z ) * Int64Mul128( y.x * z.y - y.y * z.x );
}


/// vertices of two triangles ta=234 and tb=567 split into shared and not-shared ones
struct SharedPoints
{
    int numShared = 0;           ///< number of vertices shared by ta and tb (0, 1 or 2)
    PreciseVertCoords otherA[3]; ///< first 3-numShared elements are the vertices of ta not in tb
    PreciseVertCoords otherB[3]; ///< first 3-numShared elements are the vertices of tb not in ta
};

SharedPoints findSharedPoints( const std::array<PreciseVertCoords, 8> & vs )
{
    SharedPoints res;
    int na = 0, nb = 0;
    for ( int i = 2; i < 5; ++i )
    {
        bool shared = false;
        for ( int j = 5; j < 8; ++j )
        {
            if ( vs[i].id != vs[j].id )
                continue;
            assert( vs[i].pt == vs[j].pt );
            shared = true;
        }
        if ( shared )
            ++res.numShared;
        else
            res.otherA[na++] = vs[i];
    }
    assert( res.numShared <= 2 ); // triangles sharing all three vertices are not allowed
    for ( int j = 5; j < 8; ++j )
    {
        bool shared = false;
        for ( int i = 2; i < 5; ++i )
            shared = shared || vs[i].id == vs[j].id;
        if ( !shared )
            res.otherB[nb++] = vs[j];
    }
    assert( na == nb && na + res.numShared == 3 );
    return res;
}

/// if all given points are on one side of the plane passing via vertices t0, t1, t2 then returns that side, otherwise returns nullopt
std::optional<bool> oneSideOfPlane( const std::array<PreciseVertCoords, 8> & vs, int t0, int t1, int t2, const PreciseVertCoords * pts, int numPts )
{
    assert( numPts > 0 );
    const bool side = orient3d( { vs[t0], vs[t1], vs[t2], pts[0] } );
    for ( int i = 1; i < numPts; ++i )
        if ( side != orient3d( { vs[t0], vs[t1], vs[t2], pts[i] } ) )
            return {};
    return side;
}

/// slow processing of the general case of segment intersection order, when both ta=234 and tb=567 are crossed by segment s=01,
/// and neither triangle is on one side of the other's plane
template <std::int64_t M>
bool segmentIntersectionOrderGeneral( const std::array<PreciseVertCoords, 8> & vs )
{
    // res = ( orient3d(ta,s[0])*orient3d(tb,s[1])   -   orient3d(tb,s[0])*orient3d(ta,s[1]) ) /
    //       ( orient3d(ta,s[0])-orient3d(ta,s[1]) ) * ( orient3d(tb,s[0])-orient3d(tb,s[1]) )
    const auto volumeTaOrg  = volume( vs[2].pt, vs[3].pt, vs[4].pt, vs[0].pt );
    const auto volumeTaDest = volume( vs[2].pt, vs[3].pt, vs[4].pt, vs[1].pt );
    assert( ( volumeTaOrg <= 0 && volumeTaDest >= 0 ) || ( volumeTaOrg >= 0 && volumeTaDest <= 0 ) );

    const auto volumeTbOrg  = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[0].pt );
    const auto volumeTbDest = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[1].pt );
    assert( ( volumeTbOrg <= 0 && volumeTbDest >= 0 ) || ( volumeTbOrg >= 0 && volumeTbDest <= 0 ) );

    const auto nomSimple = Int128Mul256( volumeTaOrg ) * Int128Mul256( volumeTbDest ) - Int128Mul256( volumeTbOrg ) * Int128Mul256( volumeTaDest );
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

    const auto polyTaOrg  = orient3dPoly<M>( ds[2], ds[3], ds[4], ds[0], 3 );
    const auto polyTaDest = orient3dPoly<M>( ds[2], ds[3], ds[4], ds[1], 3 );
    assert( !polyTaOrg.empty() || !polyTaDest.empty() );
    assert( polyTaOrg.empty() || polyTaDest.empty() || polyTaOrg.isPositive() != polyTaDest.isPositive() );
    const bool posTaOrg = polyTaOrg.empty() ? !polyTaDest.isPositive() : polyTaOrg.isPositive();

    const auto polyTbOrg  = orient3dPoly<M>( ds[5], ds[6], ds[7], ds[0], 3 );
    const auto polyTbDest = orient3dPoly<M>( ds[5], ds[6], ds[7], ds[1], 3 );
    assert( !polyTbOrg.empty() || !polyTbDest.empty() );
    assert( polyTbOrg.empty() || polyTbDest.empty() || polyTbOrg.isPositive() != polyTbDest.isPositive() );
    const bool posTbOrg = polyTbOrg.empty() ? !polyTbDest.isPositive() : polyTbOrg.isPositive();

    // the coefficient of zero degree is nomSimple == 0, and it is automatically excluded from nom
    auto nom = mulAs<Int128Mul256>( polyTaOrg, polyTbDest );
    nom -= mulAs<Int128Mul256>( polyTbOrg, polyTaDest );

    bool res = nom.isPositive();
    if ( posTaOrg != posTbOrg ) // denominator is negative
        res = !res;
    return res;
}

} // anonymous namespace

bool orient3d( const Vector3i & a, const Vector3i& b, const Vector3i& c )
{
    auto vhp = dot( Vector3i64mul{ a }, Vector3i64mul{ cross( Vector3i64{ b }, Vector3i64{ c } ) } );
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

bool ccwAroundLine( const PreciseVertCoords* vs )
{
    // orient3d( vs[0], vs[1], x, y ) is true iff the rotation around the line from the half-plane
    // via x to the half-plane via y is clockwise, and the three half-planes are counter-clockwise
    // iff at least two of the pairs (2,3), (3,4), (4,2) are counter-clockwise
    const bool l3 = orient3d( { vs[0], vs[1], vs[2], vs[3] } );
    const bool l4 = orient3d( { vs[0], vs[1], vs[2], vs[4] } );
    if ( l3 != l4 )
        return l4; // the pairs (2,3) and (4,2) agree, and give the answer

    return orient3d( { vs[0], vs[1], vs[4], vs[3] } ); // they disagree, so the pair (3,4) decides
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
    assert( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) );
    assert( doTriangleSegmentIntersect( { vs[5], vs[6], vs[7], vs[0], vs[1] } ) );

    // shared vertices are on both planes, so only not-shared vertices define the side of a triangle
    const auto sp = findSharedPoints( vs );
    if ( auto sideA = oneSideOfPlane( vs, 5, 6, 7, sp.otherA, 3 - sp.numShared ) )
        return *sideA == orient3d( { vs[5], vs[6], vs[7], vs[0] } ); // ta is on one side of tb's plane
    if ( auto sideB = oneSideOfPlane( vs, 2, 3, 4, sp.otherB, 3 - sp.numShared ) )
        return *sideB == orient3d( { vs[2], vs[3], vs[4], vs[1] } ); // tb is on one side of ta's plane

    // triangles ta and tb intersect one another
    return segmentIntersectionOrderGeneral<cMaxPolyDTriTri>( vs );
}

bool segmentIntersectionTriPlaneOrder( const std::array<PreciseVertCoords, 8> & vs )
{
    // s=01, ta=234, pb=567
    assert( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) );

    const bool o0 = orient3d( { vs[5], vs[6], vs[7], vs[0] } );
    if ( o0 == orient3d( { vs[5], vs[6], vs[7], vs[1] } ) )
    {
        // entire segment s is on one side of plane pb, so the line of s crosses pb either before s[0] or after s[1];
        // it is after s[1] iff s[1] is closer to pb than s[0]
        const auto volumeOrg  = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[0].pt );
        const auto volumeDest = volume( vs[5].pt, vs[6].pt, vs[7].pt, vs[1].pt );
        if ( volumeOrg != volumeDest )
            return ( volumeOrg > volumeDest ) == o0;

        // s is parallel to pb, and the perturbation of the points decides, which end of s is closer to pb
        const auto ds = getPointDegrees( vs );
        auto diff = orient3dPoly<cMaxPolyDTriPlane>( ds[5], ds[6], ds[7], ds[0], 3 );
        diff -= orient3dPoly<cMaxPolyDTriPlane>( ds[5], ds[6], ds[7], ds[1], 3 );
        return diff.isPositive() == o0;
    }

    // segment s crosses plane pb
    const auto sp = findSharedPoints( vs );
    if ( auto sideA = oneSideOfPlane( vs, 5, 6, 7, sp.otherA, 3 - sp.numShared ) )
        return *sideA == o0; // ta is on one side of pb

    // pb is infinite, so even if all its three points are on one side of ta, pb can cross s on either side of s^ta
    return segmentIntersectionOrderGeneral<cMaxPolyDTriPlane>( vs );
}

ConvertToIntVector getToIntConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double invRange = cRangeIntMax / maxDim;

    return ConvertToIntVector{ center, invRange };
}

ConvertToFloatVector getToFloatConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double range = maxDim / cRangeIntMax;

    return ConvertToFloatVector{ range, center };
}

Vector<Vector3i, VertId> computeIntCoords( const ConvertToIntVector& conv,
    const VertCoords& points, const VertBitSet* valid )
{
    MR_TIMER;
    Vector<Vector3i, VertId> res;
    res.resizeNoInit( points.size() );
    if ( valid )
    {
        BitSetParallelFor( *valid, [&]( VertId v )
        {
            res[v] = conv( points[v] );
        } );
    }
    else
    {
        ParallelFor( res, [&]( VertId v )
        {
            res[v] = conv( points[v] );
        } );
    }
    return res;
}

VertCoords computeFloatCoords( const ConvertToFloatVector& conv,
    const Vector<Vector3i, VertId>& intCoords, const VertBitSet* valid )
{
    MR_TIMER;
    VertCoords res;
    res.resizeNoInit( intCoords.size() );
    if ( valid )
    {
        BitSetParallelFor( *valid, [&]( VertId v )
        {
            res[v] = conv( intCoords[v] );
        } );
    }
    else
    {
        ParallelFor( res, [&]( VertId v )
        {
            res[v] = conv( intCoords[v] );
        } );
    }
    return res;
}

std::optional<Vector3i> findTwoSegmentsIntersection( const Vector3i& ai, const Vector3i& bi, const Vector3i& ci, const Vector3i& di )
{
    const auto ab = Vector3i64{ bi - ai };
    const auto ac = Vector3i64{ ci - ai };
    const auto ad = Vector3i64{ di - ai };
    const auto abc = cross( ab, ac );
    const auto abd = cross( ab, ad );

    if ( dot( Vector3i64mul( abc ), Vector3i64mul( abd ) ) > 0 )
        return std::nullopt; // CD is on one side of AB

    const auto cd = Vector3i64{ di - ci };
    const auto cb = Vector3i64{ bi - ci };
    const auto cda = cross( cd, -ac );
    const auto cdb = cross( cd, cb );
    if ( dot( Vector3i64mul( cda ), Vector3i64mul( cdb ) ) > 0 )
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
    FastInt128 ck = dot( Vector3i64mul( n ), Vector3i64mul( abc ) );
    FastInt128 dk = dot( Vector3i64mul( n ), Vector3i64mul( abd ) );
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
    auto abcd = dot( Vector3i64mul{ ai - di }, Vector3i64mul{ cross( Vector3i64{ bi - di }, Vector3i64{ ci - di } ) } );
    if ( abcd < 0 )
        abcd = -abcd;
    auto abce = dot( Vector3i64mul{ ai - ei }, Vector3i64mul{ cross( Vector3i64{ bi - ei }, Vector3i64{ ci - ei } ) } );
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
