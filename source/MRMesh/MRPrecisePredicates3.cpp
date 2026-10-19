#include "MRPrecisePredicates3.h"
#include "MRPrecisePredicates3Poly.h"
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

// these values were found experimentally as the largest degree of a polynomial term that must be considered in the predicates
// (the leading term of nom, or the leading term of at least one of two orient3d-polynomials for the segment's ends)
// in the tests and in a random sweep of degenerate inputs; the terms of higher degrees are not stored to save computation time,
// and if they are not enough to decide, the predicates repeat the computation with all terms (see cMaxPolyDAll)
constexpr std::int64_t cMaxPolyDTriTri   = 15'943'959; // segmentIntersectionOrder
constexpr std::int64_t cMaxPolyDTriPlane = 430'486'893; // segmentIntersectionTriPlaneOrder: it reaches the polynomial path with all three largest ids in the plane

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

/// the volumes of the tetrahedra formed by two triangles ta, tb (or planes) and the ends of segment s
struct SegmentVolumes
{
    FastInt128 taOrg, taDest, tbOrg, tbDest;
};

/// the order of intersections of segment s=01 with the planes of ta=234 and tb=567 given by exact coordinates,
/// or nullopt if the intersection points coincide exactly
std::optional<bool> segmentIntersectionOrderExact( const SegmentVolumes & v )
{
    // res = ( orient3d(ta,s[0])*orient3d(tb,s[1])   -   orient3d(tb,s[0])*orient3d(ta,s[1]) ) /
    //       ( orient3d(ta,s[0])-orient3d(ta,s[1]) ) * ( orient3d(tb,s[0])-orient3d(tb,s[1]) )
    assert( ( v.taOrg <= 0 && v.taDest >= 0 ) || ( v.taOrg >= 0 && v.taDest <= 0 ) );
    assert( ( v.tbOrg <= 0 && v.tbDest >= 0 ) || ( v.tbOrg >= 0 && v.tbDest <= 0 ) );

    const auto nomSimple = Int128Mul256( v.taOrg ) * Int128Mul256( v.tbDest ) - Int128Mul256( v.tbOrg ) * Int128Mul256( v.taDest );
    if ( nomSimple == 0 )
        return {};

    bool res = nomSimple > 0;
    assert( v.taOrg || v.taDest );
    if ( v.taOrg < v.taDest )
        res = !res;
    assert( v.tbOrg || v.tbDest );
    if ( v.tbOrg < v.tbDest )
        res = !res;
    return res;
}

/// the stages of the perturbation of the eight points of segmentIntersectionOrder and segmentIntersectionTriPlaneOrder:
/// a product of two orient3d-polynomials takes every point at most twice, with the weight at most 9 + 3 + 1 each time;
/// inlined in the degenerate cases, which then stay out of line in the callers, and their exact paths lean
MR_FORCE_INLINE PerturbationStages perturbationStages( const std::array<PreciseVertCoords, 8> & vs )
{
    constexpr std::int64_t radices[8] = { 27, 27, 27, 27, 27, 27, 27, 27 };
    return PerturbationStages( vs.data(), radices, 8 );
}

/// the order of intersections of segment s=01 with the planes of ta=234 and tb=567 when the intersection points coincide exactly,
/// resolved by simulation-of-simplicity from the terms of degrees not above M, or nullopt if they do not decide it;
/// the caller must have checked that neither triangle is on one side of the other's plane
template <std::int64_t M>
std::optional<bool> segmentIntersectionOrderDegenerate( const PerturbationStages & ps )
{
    VolumePoly<M> polyTaOrg, polyTaDest, polyTbOrg, polyTbDest;
    return ps.firstStage( [&] ( int s ) -> std::optional<bool>
    {
        // a volume changes only at the stages of its points
        auto update = [&] ( VolumePoly<M> & poly, int a, int b, int c, int d )
        {
            if ( s == 1 || ps.startsAt( a, s ) || ps.startsAt( b, s ) || ps.startsAt( c, s ) || ps.startsAt( d, s ) )
                poly = orient3dPoly<M>( ps.at( a, s ), ps.at( b, s ), ps.at( c, s ), ps.at( d, s ) );
        };
        update( polyTaOrg,  2, 3, 4, 0 );
        update( polyTaDest, 2, 3, 4, 1 );
        update( polyTbOrg,  5, 6, 7, 0 );
        update( polyTbDest, 5, 6, 7, 1 );
        if ( ( polyTaOrg.empty() && polyTaDest.empty() ) || ( polyTbOrg.empty() && polyTbDest.empty() ) )
            return {};
        assert( polyTaOrg.empty() || polyTaDest.empty() || polyTaOrg.isPositive() != polyTaDest.isPositive() );
        const bool posTaOrg = polyTaOrg.empty() ? !polyTaDest.isPositive() : polyTaOrg.isPositive();
        assert( polyTbOrg.empty() || polyTbDest.empty() || polyTbOrg.isPositive() != polyTbDest.isPositive() );
        const bool posTbOrg = polyTbOrg.empty() ? !polyTbDest.isPositive() : polyTbOrg.isPositive();

        // the sign of the leading term of nom = polyTaOrg * polyTbDest - polyTbOrg * polyTaDest, whose terms of degrees below place( s )
        // are those of the previous stage, which are zero (at the first stage: the zero-degree coefficient nomSimple == 0)
        const int nomSign = lowestTermOfProductsDiff<Int128Mul256>( polyTaOrg, polyTbDest, polyTbOrg, polyTaDest, ps.place( s ) ).second;
        if ( nomSign == 0 )
            return {};
        bool res = nomSign > 0;
        if ( posTaOrg != posTbOrg ) // denominator is negative
            res = !res;
        return res;
    }, M );
}

/// the same from the terms of degrees not above M, which are enough for most inputs, or else from all terms
template <std::int64_t M>
bool segmentIntersectionOrderDegenerate( const std::array<PreciseVertCoords, 8> & vs )
{
    const auto ps = perturbationStages( vs );
    if ( const auto res = segmentIntersectionOrderDegenerate<M>( ps ) )
        return *res;
    const auto res = segmentIntersectionOrderDegenerate<cMaxPolyDAll>( ps );
    assert( res ); // only invalid input leaves the order undecided, e.g. triangles with three shared vertices
    return res.value_or( false );
}

/// whether the leading term of the difference of the perturbed volumes of plane pb=567 with s[0] and with s[1] is positive,
/// from the terms of degrees not above M, or nullopt if they are all zero; the caller must have checked that the exact volumes are equal
template <std::int64_t M>
std::optional<bool> volumeDiffPositive( const PerturbationStages & ps )
{
    return ps.firstStage( [&] ( int s ) -> std::optional<bool>
    {
        // the difference is zero at the previous stage (at the first stage: the difference of the exact volumes),
        // so it stays zero unless one of its points moves at this stage
        if ( !ps.startsAt( 0, s ) && !ps.startsAt( 1, s ) && !ps.startsAt( 5, s ) && !ps.startsAt( 6, s ) && !ps.startsAt( 7, s ) )
            return {};
        const auto pb0 = ps.at( 5, s ), pb1 = ps.at( 6, s ), pb2 = ps.at( 7, s );
        const auto diff = orient3dPoly<M>( pb0, pb1, pb2, ps.at( 0, s ) ) - orient3dPoly<M>( pb0, pb1, pb2, ps.at( 1, s ) );
        if ( diff.empty() )
            return {};
        return diff.isPositive();
    }, M );
}

} // anonymous namespace

namespace
{

/// orient3d( a, b, c ) for the points 0, a, b, c known to be on the same plane, resolved by simulation-of-simplicity
bool orient3dDegenerate( const Vector3i & a, const Vector3i& b, const Vector3i& c )
{
    assert( dot( Vector3i64mul{ a }, Vector3i64mul{ cross( Vector3i64{ b }, Vector3i64{ c } ) } ) == 0 );

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

/// orient3d( { a, b, c, d } ) for four points known to be exactly coplanar, resolved by simulation-of-simplicity
bool orient3dDegenerate( const PreciseVertCoords & a, const PreciseVertCoords & b, const PreciseVertCoords & c, const PreciseVertCoords & d )
{
    assert( orient3dVolume( a.pt, b.pt, c.pt, d.pt ) == 0 );

    const PreciseVertCoords * vs[4] = { &a, &b, &c, &d };
    bool odd = false;
    std::array<int, 4> order = { 0, 1, 2, 3 };

    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = i + 1; j < 4; ++j )
        {
            assert( vs[order[i]]->id != vs[order[j]]->id );
            if ( vs[order[i]]->id > vs[order[j]]->id )
            {
                odd = !odd;
                std::swap( order[i], order[j] );
            }
        }
    }

    const auto & o = vs[order[3]]->pt;
    return odd != orient3dDegenerate( vs[order[0]]->pt - o, vs[order[1]]->pt - o, vs[order[2]]->pt - o );
}

/// orient3d( { a, b, c, d } ), inlined in the callers below; the degenerate case and its temporary arrays stay out of line
MR_FORCE_INLINE bool orient3dInl( const PreciseVertCoords & a, const PreciseVertCoords & b, const PreciseVertCoords & c, const PreciseVertCoords & d )
{
    // the exact answer first, the perturbation of the points is necessary only if all four points are exactly coplanar
    if ( const auto v = orient3dVolume( a.pt, b.pt, c.pt, d.pt ); v != 0 )
        return v > 0;
    return orient3dDegenerate( a, b, c, d );
}

} // anonymous namespace

bool orient3d( const Vector3i & a, const Vector3i& b, const Vector3i& c )
{
    auto vhp = dot( Vector3i64mul{ a }, Vector3i64mul{ cross( Vector3i64{ b }, Vector3i64{ c } ) } );
    if ( vhp ) return vhp > 0;
    return orient3dDegenerate( a, b, c );
}

bool orient3d( const PreciseVertCoords* vs )
{
    return orient3dInl( vs[0], vs[1], vs[2], vs[3] );
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

    // orient3dInl directly and not via a lambda: MSVC does not inline the lambda then
    const auto abcd = orient3dInl( vs[a], vs[b], vs[c], vs[d] );
    res.dIsLeftFromABC = abcd;
    const auto abce = orient3dInl( vs[a], vs[b], vs[c], vs[e] );
    if ( abcd == abce )
        return res; // segment DE is located at one side of the plane ABC

    const auto dabe = orient3dInl( vs[a], vs[b], vs[d], vs[e] );
    const auto dbce = orient3dInl( vs[b], vs[c], vs[d], vs[e] );
    if ( dabe != dbce )
        return res; // segment AC is located at one side of the plane DEB

    const auto dcae = !orient3dInl( vs[a], vs[c], vs[d], vs[e] ); // '!' is due to inverted order of a and c
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

    const SegmentVolumes v
    {
        .taOrg  = orient3dVolume( vs[2].pt, vs[3].pt, vs[4].pt, vs[0].pt ),
        .taDest = orient3dVolume( vs[2].pt, vs[3].pt, vs[4].pt, vs[1].pt ),
        .tbOrg  = orient3dVolume( vs[5].pt, vs[6].pt, vs[7].pt, vs[0].pt ),
        .tbDest = orient3dVolume( vs[5].pt, vs[6].pt, vs[7].pt, vs[1].pt )
    };
    if ( auto res = segmentIntersectionOrderExact( v ) )
        return *res;

    // the intersection points coincide exactly, and the perturbation of the points decides;
    // shared vertices are on both planes, so only not-shared vertices define the side of a triangle
    const auto sp = findSharedPoints( vs );
    if ( auto sideA = oneSideOfPlane( vs, 5, 6, 7, sp.otherA, 3 - sp.numShared ) )
        return *sideA == ( v.tbOrg != 0 ? v.tbOrg > 0 : orient3dDegenerate( vs[5], vs[6], vs[7], vs[0] ) ); // ta is on one side of tb's plane
    if ( auto sideB = oneSideOfPlane( vs, 2, 3, 4, sp.otherB, 3 - sp.numShared ) )
        return *sideB == ( v.taDest != 0 ? v.taDest > 0 : orient3dDegenerate( vs[2], vs[3], vs[4], vs[1] ) ); // tb is on one side of ta's plane

    // triangles ta and tb intersect one another
    return segmentIntersectionOrderDegenerate<cMaxPolyDTriTri>( vs );
}

bool segmentIntersectionTriPlaneOrder( const std::array<PreciseVertCoords, 8> & vs )
{
    // s=01, ta=234, pb=567
    assert( doTriangleSegmentIntersect( { vs[2], vs[3], vs[4], vs[0], vs[1] } ) );

    const auto volumeOrg  = orient3dVolume( vs[5].pt, vs[6].pt, vs[7].pt, vs[0].pt );
    const auto volumeDest = orient3dVolume( vs[5].pt, vs[6].pt, vs[7].pt, vs[1].pt );
    const bool o0 = volumeOrg  != 0 ? volumeOrg  > 0 : orient3dDegenerate( vs[5], vs[6], vs[7], vs[0] );
    const bool o1 = volumeDest != 0 ? volumeDest > 0 : orient3dDegenerate( vs[5], vs[6], vs[7], vs[1] );
    if ( o0 == o1 )
    {
        // entire segment s is on one side of plane pb, so the line of s crosses pb either before s[0] or after s[1];
        // it is after s[1] iff s[1] is closer to pb than s[0]
        if ( volumeOrg != volumeDest )
            return ( volumeOrg > volumeDest ) == o0;

        // s is parallel to pb, and the perturbation of the points decides, which end of s is closer to pb
        const auto ps = perturbationStages( vs );
        if ( const auto pos = volumeDiffPositive<cMaxPolyDTriPlane>( ps ) )
            return *pos == o0;
        // the terms of degrees above cMaxPolyDTriPlane decide
        const auto pos = volumeDiffPositive<cMaxPolyDAll>( ps );
        assert( pos ); // only invalid input makes the perturbed volumes equal, e.g. s[0] and s[1] with the same id
        return pos && *pos == o0;
    }

    // segment s crosses plane pb
    const SegmentVolumes v
    {
        .taOrg  = orient3dVolume( vs[2].pt, vs[3].pt, vs[4].pt, vs[0].pt ),
        .taDest = orient3dVolume( vs[2].pt, vs[3].pt, vs[4].pt, vs[1].pt ),
        .tbOrg  = volumeOrg,
        .tbDest = volumeDest
    };
    if ( auto res = segmentIntersectionOrderExact( v ) )
        return *res;

    // the intersection points coincide exactly, and the perturbation of the points decides
    const auto sp = findSharedPoints( vs );
    if ( auto sideA = oneSideOfPlane( vs, 5, 6, 7, sp.otherA, 3 - sp.numShared ) )
        return *sideA == o0; // ta is on one side of pb

    // pb is infinite, so even if all its three points are on one side of ta, pb can cross s on either side of s^ta
    return segmentIntersectionOrderDegenerate<cMaxPolyDTriPlane>( vs );
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
    if ( auto res = findTriangleSegmentIntersectionPreciseInt( converters.toInt( a ), converters.toInt( b ), converters.toInt( c ),
        converters.toInt( d ), converters.toInt( e ) ) )
        return converters.toFloat( *res );

    // rare case when `numSum == 0` - segment is fully inside face
    return Vector3f( ( Vector3d( d ) + Vector3d( e ) ) * 0.5 );
}

std::optional<Vector3i> findTriangleSegmentIntersectionPreciseInt(
    const Vector3i& ai, const Vector3i& bi, const Vector3i& ci,
    const Vector3i& di, const Vector3i& ei )
{
    auto abcd = dot( Vector3i64mul{ ai - di }, Vector3i64mul{ cross( Vector3i64{ bi - di }, Vector3i64{ ci - di } ) } );
    if ( abcd < 0 )
        abcd = -abcd;
    auto abce = dot( Vector3i64mul{ ai - ei }, Vector3i64mul{ cross( Vector3i64{ bi - ei }, Vector3i64{ ci - ei } ) } );
    if ( abce < 0 )
        abce = -abce;
    auto sum = abcd + abce;
    if ( sum != 0 )
        return Vector3i{ divRound( abcd * Vector3i128fast{ ei } + abce * Vector3i128fast{ di }, sum ) };
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
        return Vector3i{ Vector3d( sumVec ) / double( numSum ) };

    // rare case when `numSum == 0` - segment is fully inside face
    return std::nullopt;
}

} //namespace MR
