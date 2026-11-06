#include "MRCutTrianglePredicates.h"
#include "MRFastInt.h"
#include "MRInt64Mul128.h"
#include "MRPrecisePredicates3Poly.h"
#include "MRSparsePolynomial.h"
#include "MRTriMath.h"
#include "MRUnorientedTriangle.h"
#include "MRphmap.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <mutex>
#include <optional>

namespace MR
{

#if __GNUC__ >= 12 // false positive array-bounds/stringop warnings from GCC on the fixed-width FastInt array operations below
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

namespace
{
using detail::cMaxPolyDAll;
using detail::orient3dPoly;
using detail::orient3dVolume;
using detail::PerturbationStagesOf;
using detail::PointDegree;
using detail::VolumePoly;

// the rounded projected coordinates of an edge-triangle point are within 2^-19 integer units from the exact ones, as a ratio
// of two volumes converted in double, times a coordinate difference within 2^31; so a larger difference of two of them decides
constexpr double cApproxTol = 0x1p-17;

/// the order of two points along a projected coordinate, whose rounded values a and b decide most pairs (see cApproxTol): -1 or 1
/// as the first point comes before or after the second one, or else what exact() finds; inlined, since less is the hot path of a sort
template <typename F>
MR_FORCE_INLINE int compareCoord( double a, double b, const F & exact )
{
    if ( a < b - cApproxTol )
        return -1;
    if ( b < a - cApproxTol )
        return 1;
    return exact();
}

/// the orientation of triangle abc from the rounded projected coordinates of its vertices (see cApproxTol),
/// or nullopt if the rounding errors could change its sign
std::optional<bool> approxCcw( const Vector2d & a, const Vector2d & b, const Vector2d & c )
{
    const double ux = b.x - a.x, uy = b.y - a.y, wx = c.x - a.x, wy = c.y - a.y;
    const double l = ux * wy, r = uy * wx;
    const double area = l - r;
    // every difference is within 2^-18 + 2^-22 < 2^-17 from the exact one, which moves each product by less than 2^-17 times
    // the sum of its factors plus 2^-34; the products and their difference add rounding errors below 2^-52 of the sum of the products
    const double err = 0x1p-16 * ( std::abs( ux ) + std::abs( uy ) + std::abs( wx ) + std::abs( wy ) )
        + 0x1p-50 * ( std::abs( l ) + std::abs( r ) ) + 0x1p-30;
    if ( area > err )
        return true;
    if ( area < -err )
        return false;
    return {};
}

/// the normal of a triangle: every coordinate is below 2^30 by absolute value, as the range factor 0.99 of ConvertToIntVector
/// ensures, so every coordinate difference is below 2^31, and every component of the normal is below 2 * 2^62 = 2^63
Vector3i64 normalOf( const std::array<PreciseVertCoords, 3> & t )
{
    return dirDblArea( Vector3i64( t[0].pt ), Vector3i64( t[1].pt ), Vector3i64( t[2].pt ) );
}

/// dot( n, v ) of a normal n (see normalOf) and a vector v with the components below 2^32, as the offset of a plane through a vertex
/// or the projection of a segment on the normal: below 3 * 2^63 * 2^32 < 2^97
FastInt128 normalDot( const Vector3i64 & n, const Vector3i64 & v )
{
    return Int64Mul128( n.x ) * Int64Mul128( v.x ) + Int64Mul128( n.y ) * Int64Mul128( v.y ) + Int64Mul128( n.z ) * Int64Mul128( v.z );
}

/// the projected coordinates kx, ky of the exact direction of the line where the plane with normal n crosses the plane of triangle t,
/// cross( n, normal of t ), below 2 * 2^126 = 2^127 for the components of the normals below 2^63; both are zero if the planes
/// are parallel before the perturbation, or a normal is zero; inlined, since the slower rules of ccw and less ask it often
MR_FORCE_INLINE std::array<FastInt128, 2> lineDirectionOf( const Vector3i64 & n, const std::array<PreciseVertCoords, 3> & t, int kx, int ky )
{
    const auto line = cross( Vector3i64mul( n ), Vector3i64mul( normalOf( t ) ) );
    return { line[kx], line[ky] };
}

/// the first side of a set of sides, bit i for side i: the side of the points with these sides in common
int firstSide( int sides ) { return std::countr_zero( unsigned( sides ) ); }

bool hasVert( const std::array<PreciseVertCoords, 3> & t, VertId v )
{
    return t[0].id == v || t[1].id == v || t[2].id == v;
}

bool sameTriangle( const std::array<PreciseVertCoords, 3> & a, const std::array<PreciseVertCoords, 3> & b )
{
    return hasVert( b, a[0].id ) && hasVert( b, a[1].id ) && hasVert( b, a[2].id );
}

/// whether p and q are defined alike: crossings of one segment, in either direction, with one triangle, with its vertices in any order,
/// or crossings of the same two cuts
bool definedAlike( const PreciseCutPoint & p, const PreciseCutPoint & q )
{
    if ( p.isCrossing() || q.isCrossing() )
        return p.isCrossing() && q.isCrossing() && ( ( sameTriangle( p.tri, q.tri ) && sameTriangle( p.tri2, q.tri2 ) )
            || ( sameTriangle( p.tri, q.tri2 ) && sameTriangle( p.tri2, q.tri ) ) );
    if ( p.isVertex() || q.isVertex() || !sameTriangle( p.tri, q.tri ) )
        return false;
    const auto & s = p.seg;
    const auto & t = q.seg;
    return ( s[0].id == t[0].id && s[1].id == t[1].id ) || ( s[0].id == t[1].id && s[1].id == t[0].id );
}

/// whether the segment of p has a wedge: two distinct third vertices of its triangles
bool hasWedge( const PreciseCutPoint & p )
{
    return p.wedge[0].id.valid() && p.wedge[1].id.valid() && p.wedge[0].id != p.wedge[1].id;
}

/// whether the half-plane from line ab through y is in the angle below 180 degrees between the half-planes through x and z,
/// on the sides of planes abx and abz where the boolean's orient3d puts it
bool insideAngle( const std::array<PreciseVertCoords, 2> & ab, const PreciseVertCoords & x, const PreciseVertCoords & z, const PreciseVertCoords & y )
{
    const auto & [a, b] = ab;
    return orient3d( { a, b, x, y } ) == orient3d( { a, b, x, z } ) && orient3d( { a, b, z, y } ) == orient3d( { a, b, z, x } );
}

/// whether the triangles of the edge of q, defined alike as p, are in the angle below 180 degrees between the triangles of the edge of p:
/// each of them shared with p or in that angle, not both shared
bool nestedIn( const PreciseCutPoint & p, const PreciseCutPoint & q )
{
    int numShared = 0;
    for ( const auto & y : q.wedge )
    {
        if ( y.id == p.wedge[0].id || y.id == p.wedge[1].id )
            ++numShared;
        else if ( !insideAngle( p.seg, p.wedge[0], p.wedge[1], y ) )
            return false;
    }
    return numShared < 2;
}

#ifndef NDEBUG
/// whether every coordinate of v is below 2^30 by absolute value, for which the integer widths and the error bounds here are derived
bool inRange( const PreciseVertCoords & v )
{
    for ( int i = 0; i < 3; ++i )
        if ( v.pt[i] <= -( 1 << 30 ) || v.pt[i] >= ( 1 << 30 ) )
            return false;
    return true;
}
#endif

// ---------- simulation of simplicity for points that coincide before the perturbation ----------
// The perturbation is the one of orient3d: coordinate i of a vertex gets eps^( PointDegree::degree( i ) ), with the degrees
// from getPointDegrees, which order the terms of every polynomial by the weights of the vertices from the largest id down,
// if the radix of every vertex exceeds its weight in every term (the sum of 9, 3, 1 over its coordinates in the term, with repetitions).
// The radix of a vertex here is 1 + the sum of the bounds of its weight in the coordinates nx, ny, d (see SosFraction) of the points
// of the predicate that contain it, since a product of the coordinates of several points (as in a determinant) adds their weights:
// - a coordinate of a vertex has the weight of that vertex at most 9;
// - every term of the perturbed volume of t[0] t[1] t[2] x takes one coordinate from each row t[j] - x, all in different columns,
//   so it has the weight of every t[j] at most 9 and that of x at most 9 + 3 + 1 = 13;
// - so the coordinates nx = oa * b[k] - ob * a[k] and d = oa - ob of the crossing of segment ab with the plane of a triangle
//   (see sosPoint) have the weights of a and b at most 13 and those of the vertices of the triangle at most 9;
// - a component of the normal of a triangle has the weight of its first vertex at most 9 + 3 = 12 and of the others 9, and the offset
//   dot( n, v ) of its plane through its first vertex v that of v at most 13, so the crossing n / d of two cuts (see sosCrossing)
//   has the weights of the first vertices of the three triangles at most 13 and of their other vertices 9.
// For three points of five vertices each, the most that three points other than crossings of two cuts take, the degrees stay below
// ( 14^2 * 10^3 )^3 < 2^53, since merging two vertices into one never makes the product of their radices larger. A crossing of two cuts
// takes nine vertices, three of them those of the cut triangle, so three points take at most 3 + 3 * 6 = 21 distinct vertices, whose degrees
// may not fit in int64: those of two crossings with no common plane do, 27 * 19^2 * 14^4 * 10^8 < 2^56, and those of three crossings
// by four planes, below 2^63, but not those of two crossings with no common plane and a crossing of an edge with the triangle,
// 36 * 28^2 * 14^6 * 10^8 > 2^64, or of three crossings with no common plane, 40 * 28^2 * 14^6 * 10^12 > 2^77; so the expansion
// takes only the stages before the first one whose terms can have degrees above int64 (see SosStages), which it rarely reaches.
// The vertices are perturbed in stages (see PerturbationStagesOf), so the polynomials stay small, and the expansion stops
// at the first stage where the polynomial of the predicate is not zero.
// The coefficients of a volume are below 2^96 (sums of products of three coordinate differences below 2^31),
// those of nx, ny, d below 2^129, and those of a crossing of two cuts below 2^224 for nx, ny and 2^192 for d, as its exact numerators
// and denominator are (see crossingExact and SosVec), so 128 bits hold the former, 256 bits the latter, 512 bits their products of two
// and 768 bits those of three.

constexpr int cMaxSosVerts = 21; // the distinct vertices of three crossings of two cuts, the most that three points take

/// the stages of the perturbation of the vertices of a predicate, which end before the first one whose degrees do not fit
using SosStages = PerturbationStagesOf<cMaxSosVerts, true>;

template <typename C>
using SosPolyOf = SparsePolynomial<C, std::int64_t, cMaxPolyDAll>;
using SosPoly256 = SosPolyOf<FastInt256>;
using SosPoly512 = SosPolyOf<FastInt512>;

/// the indices of the vertices of one point among those of SosVerts: seg[0] of a vertex point; seg[0], seg[1], tri[0], tri[1], tri[2]
/// of a crossing of a segment; the vertices of the cut triangle, of tri and of tri2 of a crossing of two cuts
struct SosPointVerts
{
    std::array<int, 9> idx{};
    int num = 0;
};

/// the distinct vertices of the points of one predicate with the radices of their perturbation (see getPointDegrees): 1 + the sum of
/// the bounds of the weight of a vertex in the terms of all points with it
struct SosVerts
{
    std::array<PreciseVertCoords, cMaxSosVerts> verts;
    std::array<std::int64_t, cMaxSosVerts> radices;
    int num = 0;

    /// adds vertex v, whose weight in a term of a point is at most bound, and returns its index
    int add( const PreciseVertCoords & v, std::int64_t bound )
    {
        for ( int i = 0; i < num; ++i )
        {
            if ( verts[i].id == v.id )
            {
                radices[i] += bound;
                return i;
            }
        }
        assert( num < cMaxSosVerts );
        verts[num] = v;
        radices[num] = 1 + bound;
        return num++;
    }

    /// adds the vertices of point p of the cut triangle t, and returns their indices
    SosPointVerts add( const PreciseCutPoint & p, const std::array<PreciseVertCoords, 3> & t )
    {
        SosPointVerts res;
        auto put = [&] ( const PreciseVertCoords & v, std::int64_t bound ) { res.idx[res.num++] = add( v, bound ); };
        if ( p.isCrossing() )
        {
            for ( const auto * plane : { &t, &p.tri, &p.tri2 } )
            {
                put( ( *plane )[0], 13 );
                put( ( *plane )[1], 9 );
                put( ( *plane )[2], 9 );
            }
        }
        else if ( p.isVertex() )
            put( p.seg[0], 9 );
        else
        {
            put( p.seg[0], 13 );
            put( p.seg[1], 13 );
            for ( const auto & v : p.tri )
                put( v, 9 );
        }
        return res;
    }

    /// the stages of the perturbation of the vertices added
    SosStages stages() const
    {
        return SosStages( verts.data(), radices.data(), num );
    }
};

/// whether a point with its vertices v of the stages moves at stage s, having a vertex perturbed first at this stage
bool startsAt( const SosStages & ps, const SosPointVerts & v, int s )
{
    for ( int i = 0; i < v.num; ++i )
        if ( ps.startsAt( v.idx[i], s ) )
            return true;
    return false;
}

/// coordinate i of perturbed vertex v
template <typename C>
SosPolyOf<C> sosCoord( const PointDegree & v, int i )
{
    std::vector<typename SosPolyOf<C>::Term> terms;
    if ( v.pt[i] != 0 )
        terms.emplace_back( 0, C( std::int64_t( v.pt[i] ) ) );
    if ( v.d != 0 )
        terms.emplace_back( v.degree( i ), C( std::int64_t( 1 ) ) );
    return SosPolyOf<C>( std::move( terms ) );
}

/// the polynomial with the coefficients converted into 256 bits
SosPoly256 widen( const SosPolyOf<FastInt128> & p )
{
    std::vector<SosPoly256::Term> terms;
    terms.reserve( p.get().size() );
    for ( const auto & [deg, c] : p.get() )
        terms.emplace_back( deg, FastInt256( c ) );
    return SosPoly256( std::move( terms ) );
}

/// the product of two polynomials whose coefficients fit in 256 bits, as the bounds of the caller prove
SosPoly256 mul256( const SosPoly256 & a, const SosPoly256 & b )
{
    const auto p = mulAs<FastInt256>( a, b );
    std::vector<SosPoly256::Term> terms;
    terms.reserve( p.get().size() );
    for ( const auto & [deg, c] : p.get() )
        terms.emplace_back( deg, FastInt256( c ) );
    return SosPoly256( std::move( terms ) );
}

// the perturbed vectors of the directions of segments and lines, which only the rare points with zero exact directions need;
// Vector3 cannot hold polynomials, which have no division; the sum |p| of the absolute values of the coefficients of polynomial p
// bounds each of them and the partial sums that give it, and |p + q| <= |p| + |q|, |p * q| <= |p| * |q| (the terms dropped above
// degree cMaxPolyDAll only make it smaller); so for coordinates below 2^30 by absolute value, |p| is at most 2^30 for a perturbed
// coordinate pt[i] + eps^d, 2^31 for a difference of two, 2 * 2^31 * 2^31 = 2^63 for a component of a perturbed normal,
// and 2 * 2^63 * 2^63 = 2^127 for one of the cross product of two normals: 128 bits hold the former, and 256 bits the latter
template <typename C>
using SosVec = std::array<SosPolyOf<C>, 3>;

SosVec<FastInt128> sosCoords( const PointDegree & v )
{
    return { sosCoord<FastInt128>( v, 0 ), sosCoord<FastInt128>( v, 1 ), sosCoord<FastInt128>( v, 2 ) };
}

template <typename C>
SosVec<C> sub( const SosVec<C> & a, const SosVec<C> & b )
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

/// the cross product, with the coefficients converted into T before multiplication
template <typename T, typename C>
auto cross( const SosVec<C> & a, const SosVec<C> & b )
{
    return std::array{ mulAs<T>( a[1], b[2] ) - mulAs<T>( a[2], b[1] ), mulAs<T>( a[2], b[0] ) - mulAs<T>( a[0], b[2] ),
        mulAs<T>( a[0], b[1] ) - mulAs<T>( a[1], b[0] ) };
}

/// the vertices of a predicate perturbed at stage s (see PerturbationStagesOf), or with only the vertex of stage s perturbed (alone):
/// if the polynomial of the predicate is zero at stage s - 1, its terms of degrees below place( s ) cancel one another, and its term
/// of degree place( s ), the lowest one that can remain, is the derivative by the coordinate of that vertex with the weight 1
/// at the unperturbed points, the same when the vertices of the earlier stages are not perturbed; so if the polynomial with that vertex
/// perturbed alone has a term of that degree, it is the leading term, found with much smaller polynomials, as the crossings of two cuts
/// need, whose polynomials of all vertices grow to many thousands of terms at the later stages
struct Stage
{
    const SosStages & ps;
    int s = 1;
    bool alone = false;

    [[nodiscard]] PointDegree at( int i ) const
    {
        return !alone || ps.startsAt( i, s ) ? ps.at( i, s ) : PointDegree{ ps[i].pt, 0 };
    }
};

/// the perturbed normal of the triangle of the vertices a, b, c of the stages
SosVec<FastInt128> sosNormal( const Stage & st, int a, int b, int c )
{
    const auto x = sosCoords( st.at( a ) );
    return cross<FastInt128>( sub( sosCoords( st.at( b ) ), x ), sub( sosCoords( st.at( c ) ), x ) );
}

/// the perturbed offset dot( n, v ) of the plane with normal n through vertex v of the stages, below 3 * 2^63 * 2^30 < 2^95 (see SosVec)
SosPolyOf<FastInt128> sosOffset( const Stage & st, const SosVec<FastInt128> & n, int v )
{
    const auto x = sosCoords( st.at( v ) );
    return mulAs<FastInt128>( n[0], x[0] ) + mulAs<FastInt128>( n[1], x[1] ) + mulAs<FastInt128>( n[2], x[2] );
}

/// the perturbed crossing n / d of the cuts of the triangle t by the planes P and Q, with the vertices of t, P, Q in v of the stages
/// (see crossingExact); d is not normalized, and its sign tells the sides of the planes (see CutTrianglePredicates::crossingSideOfPlane_)
struct SosCrossing
{
    std::array<SosPoly256, 3> n;
    SosPoly256 d;
};

SosCrossing sosCrossing( const Stage & st, const SosPointVerts & v )
{
    const auto & i = v.idx;
    const auto nt = sosNormal( st, i[0], i[1], i[2] );
    const auto nP = sosNormal( st, i[3], i[4], i[5] );
    const auto nQ = sosNormal( st, i[6], i[7], i[8] );
    const auto at = widen( sosOffset( st, nt, i[0] ) );
    const auto aP = widen( sosOffset( st, nP, i[3] ) );
    const auto aQ = widen( sosOffset( st, nQ, i[6] ) );
    const auto cPQ = cross<Int128Mul256>( nP, nQ );
    const auto cQt = cross<Int128Mul256>( nQ, nt );
    const auto ctP = cross<Int128Mul256>( nt, nP );
    SosCrossing res;
    for ( int k = 0; k < 3; ++k )
    {
        res.n[k] = mul256( at, cPQ[k] ) + mul256( aP, cQt[k] ) + mul256( aQ, ctP[k] );
        res.d += mul256( widen( nt[k] ), cPQ[k] );
    }
    return res;
}

/// the lowest-degree term of the perturbed volume of the vertices a, b, c, d of the stages (see orient3dPoly) whose exact volume is zero,
/// at the first stage where it is not zero; nullopt if it stays zero, which needs repeated ids
std::optional<VolumePoly<cMaxPolyDAll>::Term> sosLeadOfVolume( const SosStages & ps, int a, int b, int c, int d )
{
    return ps.firstStage( [&] ( int s ) -> std::optional<VolumePoly<cMaxPolyDAll>::Term>
    {
        // the volume is zero at the previous stage, and changes only at the stages of its vertices
        if ( !ps.startsAt( a, s ) && !ps.startsAt( b, s ) && !ps.startsAt( c, s ) && !ps.startsAt( d, s ) )
            return {};
        const auto v = orient3dPoly<cMaxPolyDAll>( ps.at( a, s ), ps.at( b, s ), ps.at( c, s ), ps.at( d, s ) );
        if ( v.empty() )
            return {};
        return v.get().front();
    } );
}

/// a point with perturbed coordinates: ( nx / d, ny / d ) in the projection
struct SosFraction
{
    SosPoly256 nx, ny, d;
};

/// the perturbed projected point p with its vertices v of the stages; nullopt if its denominator is zero at this stage,
/// so that only the vertices perturbed at a later stage move the ends of its segment off the plane, or make the cuts of a crossing
/// of two cuts not parallel
std::optional<SosFraction> sosPoint( const PreciseCutPoint & p, const Stage & st, const SosPointVerts & v, int kx, int ky )
{
    SosFraction res;
    if ( p.isVertex() )
    {
        const auto x = st.at( v.idx[0] );
        res.nx = sosCoord<FastInt256>( x, kx );
        res.ny = sosCoord<FastInt256>( x, ky );
        res.d = SosPoly256( std::vector<SosPoly256::Term>{ { 0, FastInt256( 1 ) } } );
        return res;
    }
    if ( p.isCrossing() )
    {
        auto c = sosCrossing( st, v );
        if ( c.d.empty() )
            return {};
        res.nx = std::move( c.n[kx] );
        res.ny = std::move( c.n[ky] );
        res.d = std::move( c.d );
    }
    else
    {
        const auto a = st.at( v.idx[0] ), b = st.at( v.idx[1] );
        const auto t0 = st.at( v.idx[2] ), t1 = st.at( v.idx[3] ), t2 = st.at( v.idx[4] );
        const auto oa = orient3dPoly<cMaxPolyDAll>( t0, t1, t2, a ), ob = orient3dPoly<cMaxPolyDAll>( t0, t1, t2, b );
        // ( oa * b - ob * a ) / ( oa - ob ), as in exactCoords; the coefficients of the volumes are below 2^96 and those of the coordinates 2^30,
        // and a coefficient of a product sums at most two products of them
        res.nx = mulAs<Int128Mul256>( oa, sosCoord<FastInt128>( b, kx ) ) - mulAs<Int128Mul256>( ob, sosCoord<FastInt128>( a, kx ) );
        res.ny = mulAs<Int128Mul256>( oa, sosCoord<FastInt128>( b, ky ) ) - mulAs<Int128Mul256>( ob, sosCoord<FastInt128>( a, ky ) );
        res.d = widen( oa - ob );
        if ( res.d.empty() )
            return {};
    }
    if ( res.d.get().front().second.sign() < 0 )
    {
        res.nx = SosPoly256() - res.nx;
        res.ny = SosPoly256() - res.ny;
        res.d = SosPoly256() - res.d;
    }
    return res;
}

/// the lowest-degree term of a polynomial, as lowestTermOfSumOfProducts gives it: its degree, and the sign of its coefficient,
/// which is 0 for the zero polynomial
using LeadTerm = std::pair<std::int64_t, int>;

template <typename C>
LeadTerm leadOf( const SosPolyOf<C> & p )
{
    if ( p.empty() )
        return {};
    return { p.get().front().first, sgn( p.get().front().second ) };
}

/// the lowest-degree terms of the coordinates of perturbed vector vec( s ) at the first stage s where one of the coordinates with a bit
/// in mask is not zero; a coordinate that is still zero has a leading term of a later stage, so of a larger degree;
/// all signs are zero if the vector stays zero
template <typename F>
std::array<LeadTerm, 3> sosLeadTerms( const SosStages & ps, int mask, const F & vec )
{
    return ps.firstStage( [&] ( int s ) -> std::optional<std::array<LeadTerm, 3>>
    {
        const auto u = vec( s );
        const std::array<LeadTerm, 3> res{ leadOf( u[0] ), leadOf( u[1] ), leadOf( u[2] ) };
        for ( int i = 0; i < 3; ++i )
            if ( ( mask & ( 1 << i ) ) && res[i].second != 0 )
                return res;
        return {};
    } ).value_or( std::array<LeadTerm, 3>{} );
}

/// the vertices of the definitions of the points of a predicate that keep all of them on one plane Pi while they stay on Pi (see planeGuards)
struct PlaneGuards
{
    std::array<VertId, 9> ids;
    int num = 0;
    /// the weight (see PointDegree) of the lowest coordinate in which the normal of Pi is not zero: 1 for z, 3 for y, 9 for x;
    /// a guard moving in the other coordinates stays on Pi
    int weight = 1;
};

/// the first value that f( x, from ) gives for the perturbed projected points x[i] of pts[i] of the cut triangle tri at the stages
/// of their perturbation where all of them are defined, or the empty value if it gives none at the stages whose degrees fit in int64
/// (see SosStages); the products of the terms below degree from cancel one another in the polynomial of f,
/// which it found zero at the previous stage if from is not zero (see `from` in lowestTermOfSumOfProducts); f also gives the degree
/// of the leading term that its value comes from, so that a stage takes the value found with its vertex perturbed alone if that degree
/// is the place of the stage (see Stage), where every point keeps the sign of its denominator, which its constant term gives then;
/// the expansion starts at the first stage that perturbs a vertex of guards, as after a zero stage, if the caller proved the polynomial
/// of f zero at the stages before it (see planeGuards), and that stage takes the value found with its vertex alone if that degree is
/// the place of the stage times the weight of guards, the lowest one that a term can have there (see below)
template <int N, typename F>
auto sosFirstStage( const std::array<const PreciseCutPoint *, N> & pts, const std::array<PreciseVertCoords, 3> & tri, int kx, int ky, const F & f,
    const PlaneGuards & guards = {} )
{
    std::array<std::optional<SosFraction>, N> x;
    std::int64_t deg = 0;
    using Res = decltype( f( x, std::int64_t{}, deg ) );
    SosVerts r;
    std::array<SosPointVerts, N> verts;
    for ( int i = 0; i < N; ++i )
        verts[i] = r.add( *pts[i], tri );
    const auto ps = r.stages();
    // the first stage to expand: the stage of the guard vertex of the smallest id, after the last one if no stage perturbs a guard
    int first = 1;
    if ( guards.num > 0 )
    {
        first = ps.numStages() + 1;
        for ( int g = 0; g < guards.num; ++g )
            for ( int i = 0; i < r.num; ++i )
                if ( r.verts[i].id == guards.ids[g] )
                    for ( int s = 1; s < first; ++s )
                        if ( ps.startsAt( i, s ) )
                            first = s;
    }
    bool prevZero = first > 1; // whether all points were defined at the previous stage, and the polynomial of f zero, or the stages were skipped
    return ps.firstStage( [&] ( int s )
    {
        Res res;
        if ( s < first )
            return res;
        if ( prevZero )
        {
            // the vertex g of this stage alone gives the leading term if its degree is the lowest one that a term with g can have (see Stage):
            // place( s ), or at the first stage of guards, w * place( s ) for w = guards.weight, which fits in int64, as place( s ) times
            // the radix of g, above 9 (see SosVerts::add), does (see getPointDegrees); there f is the determinant of sosOrient, and only
            // the coordinates of g in which the normal of Pi is not zero move the points off Pi, so for Pi in x = const, the term of degree
            // place( s ), of the coordinate z, is zero: take the determinant as a polynomial in the perturbations of the coordinates
            // of the vertices of the stages up to s as independent variables; with those of g in which the normal of Pi is not zero
            // set to zero, g stays on Pi, as the other guards do (the earlier stages perturb no guard, and the vertices of the later ones
            // are not perturbed), so wherever no denominator is zero, every point is on Pi and on the cut triangle, and so all of them
            // on one line (see planeGuards), and the determinant is zero; so its product with the denominators is the zero polynomial,
            // while the denominators are not (their constant terms, of the unperturbed points, are not zero, as ok checks), so the determinant
            // is; so every term of the determinant has a variable of g in which the normal of Pi is not zero, and a degree in eps of at least
            // w * place( s ), which only the variable of weight w alone has, whose coefficient is the derivative of the determinant by it
            // at the unperturbed points, the same with g perturbed alone, whose other terms have larger degrees too; the terms of the other
            // variables may get equal degrees in eps (as y^3 and x of g), but they cancel as monomials already
            const std::int64_t lowest = s == first && guards.num > 0 ? guards.weight * ps.place( s ) : ps.place( s );
            std::array<std::optional<SosFraction>, N> alone;
            bool ok = true;
            for ( int i = 0; i < N && ok; ++i )
            {
                alone[i] = sosPoint( *pts[i], { ps, s, true }, verts[i], kx, ky );
                ok = alone[i] && alone[i]->d.get().front().first == 0;
            }
            if ( ok && ( res = f( alone, ps.place( s ), deg ) ) )
            {
                assert( deg >= lowest ); // in valid input, see above
                if ( deg == lowest )
                    return res;
            }
            res = {};
        }
        bool defined = true;
        for ( int i = 0; i < N; ++i )
        {
            // a point keeps its coordinates of the previous stage unless it has a vertex perturbed first at this one
            if ( s == first || startsAt( ps, verts[i], s ) )
                x[i] = sosPoint( *pts[i], { ps, s }, verts[i], kx, ky );
            defined = defined && x[i].has_value();
        }
        if ( defined ) // otherwise the segment of a point is still in the plane of its triangle, or the cuts of a crossing parallel
            res = f( x, prevZero ? ps.place( s ) : 0, deg );
        prevZero = defined && !res;
        return res;
    } );
}

/// the signs of the projected coordinates of a perturbed vector with these leading terms, as the sweep along kx + eta * ky
/// with eps << eta sees them: a coordinate whose leading term has a larger degree than that of the other one is taken as zero
Vector2i sweepSigns( const LeadTerm & x, const LeadTerm & y )
{
    const auto & [degX, signX] = x;
    const auto & [degY, signY] = y;
    return { signY == 0 || degX <= degY ? signX : 0, signX == 0 || degY <= degX ? signY : 0 };
}

/// whether the sweep meets the points along a direction with these signs of the projected coordinates in the order
/// of the direction: the first coordinate decides, or the second if the first is exactly zero, since eps << eta
bool sweepForward( const Vector2i & dir )
{
    return dir.x != 0 ? dir.x > 0 : dir.y > 0;
}

/// the signs of the projected coordinates of perturbed q - p, as the sweep along kx + eta * ky with eps << eta sees them
/// (see sweepSigns); nullopt if the points coincide after the perturbation too
std::optional<Vector2i> sosDirection( const PreciseCutPoint & p, const PreciseCutPoint & q, const std::array<PreciseVertCoords, 3> & tri, int kx, int ky )
{
    // the products of a numerator below 2^224 and a denominator below 2^192 fit in 512 bits, also for the crossings of two cuts;
    // two points take at most 3 + 2 * 6 = 15 distinct vertices, whose degrees fit (see cMaxSosVerts)
    return sosFirstStage<2>( { &p, &q }, tri, kx, ky, [] ( const auto & x, std::int64_t from, std::int64_t & deg ) -> std::optional<Vector2i>
    {
        const SosFraction & a = *x[0];
        const SosFraction & b = *x[1];
        // the signs and the leading degrees of the differences of the coordinates of q and p, whose common denominator is positive
        const auto dx = lowestTermOfProductsDiff<FastInt256>( b.nx, a.d, a.nx, b.d, from );
        const auto dy = lowestTermOfProductsDiff<FastInt256>( b.ny, a.d, a.ny, b.d, from );
        if ( dx.second == 0 && dy.second == 0 )
            return {}; // the vertices perturbed so far move both points alike
        // a difference that is still zero has a leading term of a later stage, so of a larger degree, and the smaller degree decides
        deg = dy.second == 0 || ( dx.second != 0 && dx.first < dy.first ) ? dx.first : dy.first;
        return sweepSigns( dx, dy );
    } );
}

/// the guards of three points of the cut triangle t with a crossing of two cuts among them, if all of them are on one plane Pi by their definitions
/// before the perturbation, as the crossings with the cuts by the triangles of a flat face of another mesh are: every point has a triangle
/// or a segment of its definition in Pi exactly (the plane of a cut of a crossing of two cuts or of the triangle that a segment crosses,
/// or the segment), or is a vertex on Pi, by the exact volumes of their vertices with a triangle of a crossing among the points that spans Pi;
/// the guards are the vertices of those triangles and segments, and the vertices on Pi; none if no such plane is found, as for points
/// without a crossing of two cuts, whose expansion stays as it was;
/// the determinant of the points is zero at every stage before the first one that perturbs a guard, so the expansion may start there,
/// as after a zero stage (see sosFirstStage): while the guards stay unperturbed, so do the triangles and the segments of the points in Pi,
/// so every point is on Pi, or undefined, with a zero row (a crossing of two cuts that are parallel at this stage, or a crossing of a segment
/// in the plane of its triangle, whose numerators are zero too in valid input: two cuts whose crossing goes to infinity fail in addPoint,
/// and a segment with equal volumes of its ends that are not zero does not cross the plane); and every defined point is on the perturbed t,
/// so the defined points are on the line where the perturbed t crosses Pi, or on none if they are parallel, while if the perturbed t is Pi,
/// the crossing of two cuts by a plane in Pi is undefined; so every term of the determinant below the place of that stage is zero;
/// without that, the triangles of three crossings on such a line with the vertices of t and of the other cuts of smaller ids
/// (a face of mesh A of a boolean cut by mesh B with a flat face split into triangles) expanded all the stages of the former in full
/// only to find zero, which took 80 seconds for one triangle (see CutTrianglePredicatesFlatFaceCrossings); and since the guards keep the points
/// on Pi also while they move along Pi, the vertex of that stage moves them off Pi first by its coordinate of the weight in PlaneGuards,
/// whose term found with that vertex alone decides if it is not zero (see sosFirstStage), also for Pi in x = const, where the term
/// of the place of that stage, of the coordinate z, is zero, and that stage was expanded with all the vertices before it, for seconds
PlaneGuards planeGuards( const std::array<const PreciseCutPoint *, 3> & pts )
{
    // the vertices of a triangle or a segment of the definition of a point that keep it on their plane, or of the vertex point itself
    struct Carrier
    {
        const PreciseVertCoords * v = nullptr;
        int num = 0;
    };
    auto carriers = [] ( const PreciseCutPoint & p ) -> std::array<Carrier, 2>
    {
        if ( p.isCrossing() )
            return { Carrier{ p.tri.data(), 3 }, Carrier{ p.tri2.data(), 3 } };
        if ( p.isVertex() )
            return { Carrier{ p.seg.data(), 1 }, Carrier{} };
        return { Carrier{ p.tri.data(), 3 }, Carrier{ p.seg.data(), 2 } };
    };
    for ( const auto * x : pts )
    {
        if ( !x->isCrossing() )
            continue;
        for ( const auto * plane : { &x->tri, &x->tri2 } )
        {
            const auto n = normalOf( *plane );
            if ( n == Vector3i64() )
                continue; // its vertices are on one line
            const auto & [a, b, c] = *plane;
            auto inPlane = [&] ( const Carrier & k )
            {
                for ( int i = 0; i < k.num; ++i )
                    if ( orient3dVolume( a.pt, b.pt, c.pt, k.v[i].pt ) != 0 )
                        return false;
                return k.num > 0;
            };
            PlaneGuards res;
            res.weight = n.z != 0 ? 1 : ( n.y != 0 ? 3 : 9 );
            bool all = true;
            for ( int i = 0; i < 3 && all; ++i )
            {
                const auto ks = carriers( *pts[i] );
                const auto it = std::find_if( ks.begin(), ks.end(), inPlane );
                all = it != ks.end();
                if ( all )
                    for ( int j = 0; j < it->num; ++j )
                        res.ids[res.num++] = it->v[j].id;
            }
            if ( all )
                return res;
        }
    }
    return {};
}

/// the sign of the doubled area of perturbed triangle pqr in the projection: the determinant of the rows ( nx, ny, d )
/// of the perturbed points, whose denominators d are positive; 0 if the points are on one line after the perturbation too,
/// or if the expansion reaches a stage whose degrees do not fit in int64, which only crossings of two cuts with many vertices give
int sosOrient( const PreciseCutPoint & p, const PreciseCutPoint & q, const PreciseCutPoint & r, const std::array<PreciseVertCoords, 3> & tri,
    int kx, int ky )
{
    const std::array<const PreciseCutPoint *, 3> pts{ &p, &q, &r };
    // also for the crossings of two cuts, with the numerators below 2^224 and the denominators below 2^192, the minors with a denominator
    // are below 2 * 2^224 * 2^192 = 2^417 and the one of the numerators below 2 * 2^224 * 2^224 = 2^449, which fit in 512 bits,
    // and their products with the elements below 2^641, whose sum fits in 768 bits
    return sosFirstStage<3>( pts, tri, kx, ky, [] ( const auto & x, std::int64_t from, std::int64_t & deg ) -> std::optional<int>
    {
        // the cofactor expansion of det( a, b, c ) along row a: three products of an element of a and a minor of b, c
        const SosFraction & a = *x[0];
        const SosFraction & b = *x[1];
        const SosFraction & c = *x[2];
        const SosPoly512 minors[3] =
        {
            mulAs<FastInt256>( b.ny, c.d ) - mulAs<FastInt256>( b.d, c.ny ),
            mulAs<FastInt256>( b.nx, c.d ) - mulAs<FastInt256>( b.d, c.nx ),
            mulAs<FastInt256>( b.nx, c.ny ) - mulAs<FastInt256>( b.ny, c.nx )
        };
        const SignedProduct<SosPoly256, SosPoly512> products[3] = { { &a.nx, &minors[0], false }, { &a.ny, &minors[1], true }, { &a.d, &minors[2], false } };
        const auto [leadDeg, sign] = lowestTermOfSumOfProducts<FastInt256, FastInt512>( products, 3, from );
        if ( sign == 0 )
            return {}; // the vertices perturbed so far keep the points on one line
        deg = leadDeg;
        return sign;
    }, planeGuards( pts ) ).value_or( 0 );
}

/// the leading terms of the perturbed normal of triangle f, or of the perturbed cross( normal of f, normal of g ) if g is given (see sosLeadTerms)
std::array<LeadTerm, 3> sosNormalLeadTerms( const std::array<PreciseVertCoords, 3> & f, const std::array<PreciseVertCoords, 3> * g, int mask )
{
    // a normal has the weight of a vertex at most 9 + 3 = 12
    SosVerts r;
    std::array<int, 6> i;
    for ( int k = 0; k < 3; ++k )
        i[k] = r.add( f[k], 12 );
    if ( !g )
    {
        const auto ps = r.stages();
        return sosLeadTerms( ps, mask, [&] ( int s ) { return sosNormal( { ps, s }, i[0], i[1], i[2] ); } );
    }
    for ( int k = 0; k < 3; ++k )
        i[3 + k] = r.add( ( *g )[k], 12 );
    const auto ps = r.stages();
    return sosLeadTerms( ps, mask, [&] ( int s )
    {
        const Stage st{ ps, s };
        return cross<Int128Mul256>( sosNormal( st, i[0], i[1], i[2] ), sosNormal( st, i[3], i[4], i[5] ) );
    } );
}

/// the sign of the perturbed ( dot( m, X ) - dot( m, R0 ) ) * d = dot( m, n ) - dot( m, R0 ) * d for the crossing X = n / d of two cuts x
/// of triangle t (see sosCrossing) and the normal m of plane R, at the first stage where it is not zero; 0 if it stays zero
int sosCrossingSide( const PreciseCutPoint & x, const std::array<PreciseVertCoords, 3> & t, const std::array<PreciseVertCoords, 3> & plane )
{
    SosVerts r;
    const auto xv = r.add( x, t );
    // the normal of the plane has the weight of R0 at most 12 and of the others 9, and its offset that of R0 at most 13;
    // the degrees of these 12 vertices at most fit, below 14^4 * 10^8 < 2^43
    const int r0 = r.add( plane[0], 13 ), r1 = r.add( plane[1], 9 ), r2 = r.add( plane[2], 9 );
    const auto ps = r.stages();
    // the lowest term of the polynomial, and its degree
    auto lead = [&] ( const Stage & st, std::int64_t from )
    {
        const auto c = sosCrossing( st, xv );
        const auto m = sosNormal( st, r0, r1, r2 );
        const auto offset = sosOffset( st, m, r0 );
        // the products of the components of m below 2^63 and of n below 2^224, and of the offset below 2^95 and d below 2^192, fit in 384 bits
        const SignedProduct<SosPolyOf<FastInt128>, SosPoly256> products[4] =
            { { &m[0], &c.n[0], false }, { &m[1], &c.n[1], false }, { &m[2], &c.n[2], false }, { &offset, &c.d, true } };
        return lowestTermOfSumOfProducts<FastInt128, FastInt256>( products, 4, from );
    };
    bool prevZero = false;
    return ps.firstStage( [&] ( int s ) -> std::optional<int>
    {
        if ( prevZero )
        {
            // the vertex of this stage alone gives the term of the place of the stage, if any (see Stage)
            if ( const auto [deg, sign] = lead( { ps, s, true }, ps.place( s ) ); sign != 0 && deg == ps.place( s ) )
                return sign;
        }
        const int sign = lead( { ps, s }, prevZero ? ps.place( s ) : 0 ).second;
        prevZero = sign == 0;
        if ( sign != 0 )
            return sign;
        return {};
    } ).value_or( 0 );
}

/// the signs of the projected coordinates of the perturbed cross( normal of f, normal of g ), as the sweep along kx + eta * ky
/// with eps << eta sees them (see sweepSigns)
Vector2i sosCrossDirection( const std::array<PreciseVertCoords, 3> & f, const std::array<PreciseVertCoords, 3> & g, int kx, int ky )
{
    const auto t = sosNormalLeadTerms( f, &g, ( 1 << kx ) | ( 1 << ky ) );
    return sweepSigns( t[kx], t[ky] );
}

/// the exact projected coordinates of a point: ( nx / d, ny / d ) with d > 0
struct ExactCoords
{
    FastInt128 nx{}, ny{}, d{};
};

/// the exact projected coordinates of any point in the integers wide enough for a crossing of two cuts: ( nx / d, ny / d ) with d > 0,
/// with the numerators below 2^224 and the denominator below 2^192, as those of a crossing (see crossingExact) and the leading coefficients
/// of the perturbed ones that give the limit of the crossing of cuts parallel before the perturbation (see SosVec) are, and the other points
/// smaller ones (see exactCoords)
struct WideCoords
{
    FastInt256 nx, ny, d;
};

/// the exact crossing X = n / d of the cuts of triangle t by the planes P and Q of two other triangles, with nt, nP, nQ their normals
/// and at, aP, aQ their offsets dot( nt, t0 ), dot( nP, P0 ), dot( nQ, Q0 ): n = at * cross( nP, nQ ) + aP * cross( nQ, nt ) + aQ * cross( nt, nP )
/// and d = dot( nt, cross( nP, nQ ) ), since then dot( nt, X ) = at, dot( nP, X ) = aP and dot( nQ, X ) = aQ; for coordinates below 2^30,
/// the components of the normals are below 2^63 (see normalOf), the offsets below 3 * 2^63 * 2^30 < 2^95, and the components of the cross
/// products of two normals below 2^127, so n is below 3 * 2^95 * 2^127 < 2^224 and d below 3 * 2^63 * 2^127 < 2^192; every vertex has
/// the weight at most 13 in their terms (see sosCrossing), unlike in the formula of X - t0, whose terms with t0 * d cancel those of a larger weight
struct CrossingExact
{
    std::array<FastInt256, 3> n;
    FastInt256 d;
};

CrossingExact crossingExact( const std::array<PreciseVertCoords, 3> & t, const std::array<PreciseVertCoords, 3> & P,
    const std::array<PreciseVertCoords, 3> & Q )
{
    const auto nt = normalOf( t ), nP = normalOf( P ), nQ = normalOf( Q );
    const FastInt128 at = normalDot( nt, Vector3i64( t[0].pt ) ), aP = normalDot( nP, Vector3i64( P[0].pt ) ), aQ = normalDot( nQ, Vector3i64( Q[0].pt ) );
    const auto cPQ = cross( Vector3i64mul( nP ), Vector3i64mul( nQ ) );
    const auto cQt = cross( Vector3i64mul( nQ ), Vector3i64mul( nt ) );
    const auto ctP = cross( Vector3i64mul( nt ), Vector3i64mul( nP ) );
    CrossingExact res;
    for ( int k = 0; k < 3; ++k )
    {
        res.n[k] = Int128Mul256( at ) * Int128Mul256( cPQ[k] ) + Int128Mul256( aP ) * Int128Mul256( cQt[k] ) + Int128Mul256( aQ ) * Int128Mul256( ctP[k] );
        res.d += Int128Mul256( FastInt128( nt[k] ) ) * Int128Mul256( cPQ[k] );
    }
    return res;
}

/// a crossing of two cuts with its exact data
struct CrossingData
{
    WideCoords exact; // of the crossing, or of its limit if the cuts are parallel before the perturbation
    // n and d of the crossing, or their leading coefficients at the degree of the limit (see addCrossing_); all zero for the limit
    // in a triangle with a zero normal, or for a failed point
    CrossingExact parts;
    int dSign = 0; // the sign of the perturbed d (see crossingExact), which tells the sides of the planes (see crossingSideOfPlane_)
};

/// M = dot( m, n ) - offset * d for the parts of a crossing of two cuts (see CrossingData) and the plane with normal m and offset dot( m, R0 ),
/// whose sign times that of d is the side of the plane (see crossingSideOfPlane_); with the components of m below 2^63 and of n below 2^224,
/// and the offset below 2^95 and d below 2^192, M is below 2^289
FastInt<384> planeValue( const CrossingExact & x, const Vector3i64 & m, const FastInt128 & offset )
{
    return FastInt<384>( x.n[0] * m.x + x.n[1] * m.y + x.n[2] * m.z ) - offset * x.d;
}

/// a point with what the predicates take from its definition most often
struct CutPoint
{
    PreciseCutPoint def;
    ExactCoords exact; // found once in addPoint, since a sort compares the same points many times where their rounded coordinates cannot decide
    int sides = 0; // bit i is set if the point is on side i of the triangle, from vertex i to vertex i+1
    int crossing = -1; // the index of the data of a crossing of two cuts, whose exact coordinates are not in exact
};

/// whether point x, not a vertex, is on the plane of triangle t by its definition: a crossing of a side with t, a crossing of an edge of t
/// with the cut triangle, or a crossing of two cuts, one of them by t
bool onPlane( const std::array<PreciseVertCoords, 3> & t, const CutPoint & x )
{
    if ( x.def.isCrossing() )
        return sameTriangle( t, x.def.tri ) || sameTriangle( t, x.def.tri2 );
    return x.sides != 0 ? sameTriangle( t, x.def.tri ) : hasVert( t, x.def.seg[0].id ) && hasVert( t, x.def.seg[1].id );
}

/// the exact coordinates of point x in the integers of a crossing of two cuts
WideCoords wideOf( const CutPoint & x, const std::vector<CrossingData> & crossings )
{
    if ( x.crossing >= 0 )
        return crossings[x.crossing].exact;
    return { FastInt256( x.exact.nx ), FastInt256( x.exact.ny ), FastInt256( x.exact.d ) };
}

/// -1, 0, 1 as the exact projected coordinate y (x if not y) of a is less than, equal to, greater than that of b;
/// the products of the numerators below 2^224 and the denominators below 2^192 fit in 512 bits
int compareWide( const WideCoords & a, const WideCoords & b, bool y )
{
    const auto l = ( y ? a.ny : a.nx ) * b.d;
    const auto r = ( y ? b.ny : b.nx ) * a.d;
    return l < r ? -1 : ( r < l ? 1 : 0 );
}

/// the sign of the determinant of the rows ( nx, ny, d ) of the exact projected points, which is that of the doubled area of pqr;
/// the minors with a denominator below 2 * 2^224 * 2^192 = 2^417 and the one of the numerators below 2 * 2^224 * 2^224 = 2^449 fit
/// in 512 bits, and the determinant, the sum of three products below 2^641, in 768 bits
int orientWide( const WideCoords & p, const WideCoords & q, const WideCoords & r )
{
    const auto det = ( q.ny * r.d - q.d * r.ny ) * p.nx - ( q.nx * r.d - q.d * r.nx ) * p.ny + ( q.nx * r.ny - q.ny * r.nx ) * p.d;
    return det.sign();
}

/// the exact projected coordinates of the crossing of segment ab with the plane of a triangle, where oa and ob are the volumes
/// of the triangle with a and b (see addPoint)
ExactCoords exactCoords( const Vector3i & a, const Vector3i & b, const FastInt128 & oa, const FastInt128 & ob, int kx, int ky )
{
    // ( oa * b - ob * a ) / ( oa - ob ), where oa and ob have different signs (or one of them is zero);
    // by Hadamard's inequality |oa|, |ob| < 3^1.5 * 2^93 < 0.65 * 2^96 for coordinate differences below 2^31 (and the leading
    // coefficients of perturbed volumes are smaller), so the numerators are below ( |oa| + |ob| ) * 2^30 < 1.3 * 2^126,
    // and the denominator below 1.3 * 2^96
    ExactCoords res{
        oa * FastInt128( b[kx] ) - ob * FastInt128( a[kx] ),
        oa * FastInt128( b[ky] ) - ob * FastInt128( a[ky] ),
        oa - ob };
    if ( res.d < 0 )
    {
        res.nx = -res.nx;
        res.ny = -res.ny;
        res.d = -res.d;
    }
    return res;
}

/// -1, 0, 1 as the exact projected coordinate y (x if not y) of a is less than, equal to, greater than that of b
MR_FORCE_INLINE int compareExactCoord( const ExactCoords & a, const ExactCoords & b, bool y )
{
    // the products of a numerator below 1.3 * 2^126 and a denominator below 1.3 * 2^96 (see exactCoords) fit in 256 bits
    const auto l = Int128Mul256( y ? a.ny : a.nx ) * Int128Mul256( b.d );
    const auto r = Int128Mul256( y ? b.ny : b.nx ) * Int128Mul256( a.d );
    return l < r ? -1 : ( r < l ? 1 : 0 );
}

} // anonymous namespace

struct CutTrianglePredicates::Impl
{
    Vector<CutPoint, VertId> points;

    /// the data of the crossings of two cuts, which few points are
    std::vector<CrossingData> crossings;

    /// the directions from p to q of the points that coincide before the perturbation, found by the slower rules, while a sweep compares
    /// the same points often: from a side (the order of two crossings of it, or a plane line through a point on it), or by expanding
    /// the perturbation; the key is ( p << 32 ) | q for p < q, and a zero direction marks points that coincide after the perturbation too;
    /// a sweep sorts the points in parallel, hence the map with a mutex in every submap
    phmap::parallel_flat_hash_map_m<std::uint64_t, Vector2i> dirs;

    /// the orientations of the triangles with the vertices sorted by ids that the slower rules decide: 1 if counterclockwise,
    /// -1 otherwise; the sweep asks about the same triangles often; ccw may be called concurrently, but one mutex is enough,
    /// since the sweep asks about the triangles from one thread
    HashMap<UnorientedTriangle, int> orients;
    std::mutex orientsMutex;

    /// whether the maps above may have entries, since clearing the empty submaps of the parallel map takes a lock each
    std::atomic<bool> cached{ false };

    /// whether the maps may have entries found by the shifts of the points defined alike, which depend on the other points defined alike
    std::atomic<bool> shifted{ false };

    std::atomic<bool> failed{ false };

    /// forgets the orders and orientations found by the slower rules
    void clearCaches()
    {
        shifted.store( false, std::memory_order_relaxed );
        if ( !cached.exchange( false, std::memory_order_relaxed ) )
            return;
        dirs.clear();
        orients.clear();
    }
};

/// the shift of the edge of a point defined alike with others (see CutTrianglePredicates): x and y are the projected coordinates
/// of the direction in which the point moves, times scale > 0, which is the same for all points defined alike
struct CutTrianglePredicates::Shift
{
    FastInt256 x, y;
    FastInt128 scale;
};

CutTrianglePredicates::CutTrianglePredicates() : impl_( std::make_unique<Impl>() )
{
}

CutTrianglePredicates::CutTrianglePredicates( CutTrianglePredicates && ) noexcept = default;
CutTrianglePredicates & CutTrianglePredicates::operator =( CutTrianglePredicates && ) noexcept = default;
CutTrianglePredicates::~CutTrianglePredicates() = default;

bool CutTrianglePredicates::failed() const
{
    return impl_->failed.load( std::memory_order_relaxed );
}

void CutTrianglePredicates::fail_() const
{
    impl_->failed.store( true, std::memory_order_relaxed );
}

void CutTrianglePredicates::reset( const std::array<PreciseVertCoords, 3> & tri )
{
    assert( std::all_of( tri.begin(), tri.end(), inRange ) );
    tri_ = tri;
    approx_.clear();
    impl_->points.clear();
    impl_->crossings.clear();
    impl_->clearCaches();
    impl_->failed.store( false, std::memory_order_relaxed );
    if ( tri[0].id == tri[1].id || tri[1].id == tri[2].id || tri[2].id == tri[0].id )
        fail_(); // repeated ids

    normal_ = normalOf( tri );
    // drop the axis most aligned with the normal; order the two kept axes so that the projection keeps the orientation around the normal
    int dropAx = 0;
    for ( int i = 1; i < 3; ++i )
        if ( std::abs( normal_[i] ) > std::abs( normal_[dropAx] ) )
            dropAx = i;
    int dropSign = sgn( normal_[dropAx] );
    if ( dropSign == 0 && !failed() )
    {
        // the vertices are on one line before the perturbation: drop the axis of the leading term of the perturbed normal;
        // dropping any component whose leading term is not zero gives a bijective projection, oriented by the sign of that term,
        // so of the components with leading terms of equal degree (as a perturbed z coordinate gives to n_x and n_y)
        // the first one is as good as any
        const auto n = sosNormalLeadTerms( tri, nullptr, 7 );
        std::int64_t leadDeg = 0;
        for ( int i = 0; i < 3; ++i )
            if ( const auto [deg, sign] = n[i]; sign != 0 && ( dropSign == 0 || deg < leadDeg ) )
            {
                dropAx = i;
                dropSign = sign;
                leadDeg = deg;
            }
        assert( dropSign != 0 ); // the perturbed points of distinct ids are not on one line
    }
    kx_ = ( dropAx + 1 ) % 3;
    ky_ = ( dropAx + 2 ) % 3;
    if ( dropSign < 0 )
        std::swap( kx_, ky_ );

    // the directions of the sides, which the rules for the points on a side take often; their signs are found without the branches of sgn,
    // which mispredict on random triangles; only the perturbation gives the direction of a side along the dropped axis, as a side
    // of a triangle with a zero exact normal can be, and nullopt needs repeated ids
    for ( int i = 0; i < 3; ++i )
    {
        const auto & a = tri[i];
        const auto & b = tri[( i + 1 ) % 3];
        auto & dir = sideDirs_[i];
        const std::int64_t dx = std::int64_t( b.pt[kx_] ) - a.pt[kx_], dy = std::int64_t( b.pt[ky_] ) - a.pt[ky_];
        dir.x = int( dx > 0 ) - int( dx < 0 );
        dir.y = int( dy > 0 ) - int( dy < 0 );
        if ( dir == Vector2i() )
            dir = sosDirection( { .seg = { a } }, { .seg = { b } }, tri, kx_, ky_ ).value_or( Vector2i() );
    }
}

void CutTrianglePredicates::addPoint( VertId v, const PreciseCutPoint & p )
{
    assert( inRange( p.seg[0] ) && ( p.isVertex() || ( inRange( p.seg[1] ) && std::all_of( p.tri.begin(), p.tri.end(), inRange ) ) ) );
    assert( !hasWedge( p ) || ( inRange( p.wedge[0] ) && inRange( p.wedge[1] ) ) );
    assert( !p.isCrossing() || std::all_of( p.tri2.begin(), p.tri2.end(), inRange ) );
    auto & points = impl_->points;
    if ( points.size() <= v )
    {
        points.resize( v + 1 );
        approx_.resize( v + 1 );
    }
    auto & pt = points[v];
    // the point is set again: forget the orders found for its old definition; or a point with a wedge may change the shifts
    // of the points defined alike as it, and so the orders found by them
    if ( pt.def.seg[0].id.valid() || pt.def.isCrossing() || ( impl_->shifted.load( std::memory_order_relaxed ) && hasWedge( p ) ) )
        impl_->clearCaches();
    pt.def = p;
    if ( p.isCrossing() )
    {
        addCrossing_( v, p );
        return;
    }
    pt.crossing = -1;
    auto corner = [&] ( const PreciseVertCoords & x )
    {
        for ( int i = 0; i < 3; ++i )
            if ( tri_[i].id == x.id )
            {
                assert( tri_[i].pt == x.pt ); // an id stands for one vertex with the same coordinates in every point
                return i;
            }
        return -1;
    };
    if ( p.isVertex() )
    {
        const int i = corner( p.seg[0] );
        if ( i < 0 )
            fail_(); // not a vertex of the triangle, so on none of its sides, while commonLine_ takes the lines through a vertex
        pt.sides = i >= 0 ? ( 1 << i ) | ( 1 << ( ( i + 2 ) % 3 ) ) : 0;
        pt.exact = { p.seg[0].pt[kx_], p.seg[0].pt[ky_], 1 };
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        return;
    }
    const int i = corner( p.seg[0] ), j = corner( p.seg[1] );
    pt.sides = i >= 0 && j >= 0 ? 1 << ( j == ( i + 1 ) % 3 ? i : j ) : 0;
    // the volumes of the triangle of the point with the ends of its segment; if both are zero, the leading coefficients
    // of the perturbed volumes, with zero for the one of the larger degree
    FastInt128 oa = 0, ob = 0;
    bool crosses = false;
    if ( p.seg[0].id == p.seg[1].id || hasVert( p.tri, p.seg[0].id ) || hasVert( p.tri, p.seg[1].id )
        || p.tri[0].id == p.tri[1].id || p.tri[1].id == p.tri[2].id || p.tri[2].id == p.tri[0].id )
    {
        // repeated ids, also an end of the segment at a vertex of its triangle, which orient3d does not take
        pt.sides = 0;
    }
    else if ( pt.sides == 0 && !sameTriangle( p.tri, tri_ ) )
    {
        // off the sides, a segment must be an edge of another triangle crossing this one, as ccwBySides_ and commonLine_ take it;
        // a segment with a single end at a corner is neither
    }
    else
    {
        oa = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
        ob = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
        if ( oa == 0 && ob == 0 )
        {
            // the segment is in the plane, and only the perturbation moves its ends off it: as eps -> 0, the crossing tends to
            // the point where the leading terms of the perturbed volumes cancel, or to the end whose volume has the leading term
            // of the larger degree, since the other volume is infinitely larger then
            SosVerts r;
            const auto pv = r.add( p, tri_ );
            const auto ps = r.stages();
            const auto va = sosLeadOfVolume( ps, pv.idx[2], pv.idx[3], pv.idx[4], pv.idx[0] );
            const auto vb = sosLeadOfVolume( ps, pv.idx[2], pv.idx[3], pv.idx[4], pv.idx[1] );
            assert( va && vb ); // the perturbed volumes of distinct vertices are not zero
            const auto & [degA, leadA] = *va;
            const auto & [degB, leadB] = *vb;
            // the leading terms decide as orient3d does, which ties the degrees above to its perturbation
            assert( ( leadA > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } ) );
            assert( ( leadB > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } ) );
            oa = degA <= degB ? leadA : 0;
            ob = degB <= degA ? leadB : 0;
            crosses = ( leadA > 0 ) != ( leadB > 0 );
        }
        else
        {
            // an end exactly on the plane is on the side the perturbation puts it on
            const bool sa = oa != 0 ? oa > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } );
            const bool sb = ob != 0 ? ob > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } );
            crosses = sa != sb;
        }
    }
    if ( !crosses )
    {
        // repeated ids, a segment off the sides with another triangle, or the segment does not cross the plane, also after the perturbation
        fail_();
        oa = 1; // any point of the segment, since the answers are arbitrary now
        ob = -1;
    }
    // that the crossing is inside its triangle stays the caller's precondition, checked in Debug only, since it takes several orient3d
    assert( !crosses || doTriangleSegmentIntersect( { p.tri[0], p.tri[1], p.tri[2], p.seg[0], p.seg[1] } ) );
    pt.exact = exactCoords( p.seg[0].pt, p.seg[1].pt, oa, ob, kx_, ky_ );
    // the point is a + s * ( b - a ), s = oa / ( oa - ob ), where oa and ob have different signs (or one of them is zero)
    const Vector3d a( p.seg[0].pt ), b( p.seg[1].pt );
    const double da = toDouble( oa );
    const double s = da / ( da - toDouble( ob ) );
    approx_[v] = { a[kx_] + s * ( b[kx_] - a[kx_] ), a[ky_] + s * ( b[ky_] - a[ky_] ) };
}

void CutTrianglePredicates::addCrossing_( VertId v, const PreciseCutPoint & p )
{
    auto & pt = impl_->points[v];
    pt.sides = 0;
    if ( pt.crossing < 0 )
    {
        pt.crossing = int( impl_->crossings.size() );
        impl_->crossings.emplace_back();
    }
    auto & c = impl_->crossings[pt.crossing];
    c = {};
    auto distinct = [] ( const std::array<PreciseVertCoords, 3> & t )
    {
        return t[0].id.valid() && t[1].id.valid() && t[2].id.valid() && t[0].id != t[1].id && t[1].id != t[2].id && t[2].id != t[0].id;
    };
    // a triangle with two corners of this triangle cuts it along a side, where a crossing is on the side, and this triangle does not cut itself
    auto alongSide = [&] ( const std::array<PreciseVertCoords, 3> & t )
    {
        return int( hasVert( t, tri_[0].id ) ) + int( hasVert( t, tri_[1].id ) ) + int( hasVert( t, tri_[2].id ) ) >= 2;
    };
    // the crossing is another point also after the perturbation if both triangles have a corner of this triangle, which they cross at,
    // or two common vertices, whose edge they cross where it crosses this triangle (and the same triangle if three, while distinct
    // triangles in one plane give parallel cuts, which cross at the limit of their perturbed crossing, see below)
    const int numCommon = int( hasVert( p.tri, p.tri2[0].id ) ) + int( hasVert( p.tri, p.tri2[1].id ) ) + int( hasVert( p.tri, p.tri2[2].id ) );
    auto atCorner = [&]
    {
        for ( const auto & x : tri_ )
            if ( hasVert( p.tri, x.id ) && hasVert( p.tri2, x.id ) )
                return true;
        return false;
    };
    bool valid = distinct( p.tri ) && distinct( p.tri2 ) && numCommon < 2 && !alongSide( p.tri ) && !alongSide( p.tri2 ) && !atCorner();
    if ( valid )
        c.parts = crossingExact( tri_, p.tri, p.tri2 );
    if ( valid && c.parts.d.sign() != 0 )
    {
        c.dSign = c.parts.d.sign();
        c.exact = { c.parts.n[kx_], c.parts.n[ky_], c.parts.d };
    }
    else if ( valid )
    {
        // the cuts are parallel before the perturbation, and only the perturbation makes them cross: as eps -> 0, the crossing tends to the point
        // where the leading term of the denominator meets the terms of the numerators of the same degree, as no terms of the numerators of a lower
        // degree are there unless the crossing goes away to infinity, out of the triangle, which is invalid
        c.parts = {};
        valid = false;
        SosVerts r;
        const auto vs = r.add( p, tri_ );
        const auto ps = r.stages();
        (void)ps.firstStage( [&] ( int s )
        {
            // the denominator and the numerators are zero at the previous stage (see below), so the vertex of this stage alone gives
            // their terms of the place of the stage, if the denominator has one (see Stage)
            for ( const bool alone : { true, false } )
            {
                if ( alone && s == 1 )
                    continue; // the same as all vertices perturbed
                const auto x = sosCrossing( { ps, s, alone }, vs );
                if ( x.d.empty() )
                {
                    if ( alone )
                        continue;
                    // still parallel at this stage: the crossing goes to infinity if a numerator is not zero, since every term
                    // of the denominator at a later stage has a degree of at least the place of that stage, above every term found now
                    return !x.n[kx_].empty() || !x.n[ky_].empty();
                }
                const auto & [deg, cd] = x.d.get().front();
                if ( alone && deg != ps.place( s ) )
                    continue;
                // the coefficients of the terms of degree deg of all components of n, by the axes, zero for a component without one
                const int kz = 3 - kx_ - ky_;
                std::array<FastInt256, 3> n;
                for ( int k = 0; k < 3; ++k )
                {
                    const auto & num = x.n[k];
                    if ( num.empty() )
                        continue;
                    const auto & [numDeg, numCoef] = num.get().front();
                    if ( numDeg < deg && k != kz )
                        return true; // at infinity: stays invalid
                    assert( numDeg >= deg || normal_[kz] == 0 ); // see below
                    if ( numDeg == deg )
                        n[k] = numCoef;
                }
                c.dSign = cd.sign();
                c.exact = { n[kx_], n[ky_], cd };
                // the leading coefficients give the side of a plane at the limit, as the exact parts at an exact crossing (see crossingSideOfPlane_),
                // if the exact normal of this triangle is not zero: n[kx] and n[ky] have no terms below deg (see above), and n[kz] has none either,
                // since dot( nt, n ) = at * d identically (see crossingExact), so nt[kz] * n[kz] = at * d - nt[kx] * n[kx] - nt[ky] * n[ky] has none,
                // while nt[kz] has the constant term normal_[kz], the largest component of the exact normal, not zero then; so for any plane R,
                // M = dot( m, n ) - dot( m, R0 ) * d with the perturbed normal m and vertex R0 has no terms below deg, and its term of degree deg
                // is M0 = dot( m, n0 ) - dot( m, R0 ) * d0 with the unperturbed m and R0 and the coefficients n0 and d0 of n and d at deg, since
                // the perturbation adds only terms of positive degrees to m and R0; with the vertices of R added, as in sosCrossingSide,
                // the degrees of the terms change, but neither their order, by the weights of the vertices from the largest id down
                // (see getPointDegrees), nor their coefficients; so M0, if not zero, has the sign of M that sosCrossingSide finds, and the side
                // follows with dSign as at an exact crossing; n0 and d0 are below 2^224 and 2^192, as all the coefficients are (see SosVec),
                // so M0 is below 2^289 as there; with a zero normal, nt[kz] has no constant term, and the parts stay zero, for sosCrossingSide
                if ( normal_[kz] != 0 )
                    c.parts = { n, cd };
                valid = true;
                return true;
            }
            return false;
        } );
    }
    if ( !valid )
    {
        // repeated ids, a triangle taken twice, along a side or this one, cuts crossing at a corner or at the crossing of a common edge, or cuts
        // that do not cross also after the perturbation, or whose crossing goes to infinity
        fail_();
        c.dSign = 1;
        c.exact = { FastInt256( tri_[0].pt[kx_] ), FastInt256( tri_[0].pt[ky_] ), FastInt256( 1 ) }; // any point, since the answers are arbitrary now
    }
    if ( c.exact.d.sign() < 0 )
    {
        c.exact.nx = -c.exact.nx;
        c.exact.ny = -c.exact.ny;
        c.exact.d = -c.exact.d;
    }
    // each of the conversions is within 2^-53 of its value, and so is the quotient, so the rounded coordinates below 2^30 are within
    // 2^30 * 3 * 2^-53 < 2^-21 integer units from the exact ones
    const double d = toDouble( c.exact.d );
    approx_[v] = { toDouble( c.exact.nx ) / d, toDouble( c.exact.ny ) / d };
#ifndef NDEBUG
    // that the crossing is inside the triangle stays the caller's precondition, checked in Debug only; it may be on a side in the limit
    // of parallel cuts, whose crossing after the perturbation is inside
    if ( valid )
    {
        auto corner = [&] ( int i ) { return WideCoords{ FastInt256( tri_[i].pt[kx_] ), FastInt256( tri_[i].pt[ky_] ), FastInt256( 1 ) }; };
        for ( int i = 0; i < 3; ++i )
            assert( orientWide( corner( i ), corner( ( i + 1 ) % 3 ), c.exact ) >= 0 );
        // the parts are on the plane of this triangle, the limit as the exact crossing, as dot( nt, n ) = at * d gives (see above)
        assert( planeValue( c.parts, normal_, normalDot( normal_, Vector3i64( tri_[0].pt ) ) ).sign() == 0 );
    }
#endif
}

bool CutTrianglePredicates::isCrossing_( VertId v ) const
{
    return impl_->points[v].crossing >= 0;
}

/// the order of points p and q along the projected coordinate y if yFirst (x otherwise), whose rounded values cannot tell it,
/// and along the other coordinate where the points are level along the first one: -1 or 1 as p comes before or after q along
/// the first coordinate, -2 or 2 along the second one, 0 if the points coincide before the perturbation
int CutTrianglePredicates::compareExact_( VertId p, VertId q, bool yFirst ) const
{
    if ( isCrossing_( p ) || isCrossing_( q ) )
    {
        // a crossing of two cuts takes wider integers, and is rare enough to decide both coordinates by them
        const auto a = wideOf( impl_->points[p], impl_->crossings );
        const auto b = wideOf( impl_->points[q], impl_->crossings );
        if ( const int c = compareWide( a, b, yFirst ) )
            return c;
        return 2 * compareWide( a, b, !yFirst );
    }
    const auto & a = impl_->points[p].exact;
    const auto & b = impl_->points[q].exact;
    if ( const int c = compareExactCoord( a, b, yFirst ) )
        return c;
    // the exact coordinates decide the second one too where its rounded values cannot
    const int k = yFirst ? 0 : 1;
    return 2 * compareCoord( approx_[p][k], approx_[q][k], [&] { return compareExactCoord( a, b, !yFirst ); } );
}

bool CutTrianglePredicates::less( VertId p, VertId q ) const
{
    if ( p == q )
        return false;
    // along x, then along y
    if ( const int c = compareCoord( approx_[p].x, approx_[q].x, [&] { return compareExact_( p, q, false ); } ) )
        return c < 0;
    if ( const auto dir = coincidentDirection_( p, q ) )
        return sweepForward( *dir );
    return p < q; // the failure: points defined alike, not told apart by their shifts, ordered by their ids
}

/// the signs of the projected coordinates of perturbed q - p for points p and q that coincide before the perturbation,
/// as the sweep sees them (see sweepSigns), or of the difference of their shifts for points defined alike;
/// nullopt and the failure if they coincide after the perturbation and the shifts too
std::optional<Vector2i> CutTrianglePredicates::coincidentDirection_( VertId p, VertId q ) const
{
    const auto & a = impl_->points[p];
    const auto & b = impl_->points[q];
    // a vertex and another point of a side from it, a crossing strictly inside the side or the vertex at its other end: q - p goes
    // along side i in the direction that reset found if it goes from vertex i or to vertex i+1; the same vertex twice has two sides in common
    if ( const int common = a.sides & b.sides; std::has_single_bit( unsigned( common ) ) && ( a.def.isVertex() || b.def.isVertex() ) )
    {
        const int i = firstSide( common );
        const bool forward = a.def.isVertex() ? a.def.seg[0].id == tri_[i].id : b.def.seg[0].id != tri_[i].id;
        return forward ? sideDirs_[i] : -sideDirs_[i];
    }
    // the other pairs by the direction of the perturbed points, which the slower rules find and the cache keeps: two crossings of one side
    // as the boolean's cut orders them along it (see directionFromSide_), and the same point twice by the shifts of its edges
    const auto dir = perturbedDirection_( p, q, definedAlike( a.def, b.def ) );
    if ( !dir )
        fail_(); // the points coincide after the perturbation and the shifts too
    return dir;
}

std::optional<Vector2i> CutTrianglePredicates::directionFromSide_( VertId p, VertId q ) const
{
    const auto & a = impl_->points[p];
    const auto & b = impl_->points[q];
    if ( ( a.sides & b.sides ) != 0 && !a.def.isVertex() && !b.def.isVertex() )
    {
        // two crossings of one side with other triangles: q - p goes along the side, in the order the boolean's cut puts them,
        // which segmentIntersectionOrder finds slowly here, since the crossings coincide before the perturbation
        const auto & s = a.def.seg;
        const bool aFirst = segmentIntersectionOrder( { s[0], s[1], a.def.tri[0], a.def.tri[1], a.def.tri[2], b.def.tri[0], b.def.tri[1], b.def.tri[2] } );
        // forward along side i iff a comes first along a segment from vertex i, or last along one from vertex i+1
        const int i = firstSide( a.sides );
        return aFirst == ( s[0].id == tri_[i].id ) ? sideDirs_[i] : -sideDirs_[i];
    }
    // a point on a side and a point not on it, both on one plane line: the latter is strictly on the left of the side,
    // so the direction from the former to it is the direction of the line, signed to point to the left of the side
    const int onlyA = a.sides & ~b.sides;
    const int onlyB = b.sides & ~a.sides;
    if ( onlyA == 0 && onlyB == 0 )
        return {};
    // no line through a vertex: a vertex v exactly at a crossing of segment cd is on the line of cd, so plane vcd has a zero normal,
    // and exactly at a crossing of a side with a triangle of v it is on that side, so this triangle is flat: no line has a direction then
    const auto l = commonLine_( p, q, false );
    if ( !l || l->side >= 0 )
        return {};
    const auto [lx, ly] = lineDirectionOf( normal_, l->plane, kx_, ky_ );
    const Vector2i dir( sgn( lx ), sgn( ly ) );
    if ( dir == Vector2i() )
        return {}; // the line has no exact direction: the planes are parallel before the perturbation, or a normal is zero
    for ( int i = 0; i < 3; ++i )
    {
        const int bit = 1 << i;
        if ( ( ( onlyA | onlyB ) & bit ) == 0 )
            continue;
        // the sign of the cross product of the directions of the side and of the line, with factors within 2^31 and 2^127
        const auto s = Vector3i64( tri_[( i + 1 ) % 3].pt ) - Vector3i64( tri_[i].pt );
        const auto c = Int128Mul256( s[kx_] ) * Int128Mul256( ly ) - Int128Mul256( s[ky_] ) * Int128Mul256( lx );
        if ( const int cs = c.sign() )
            return ( cs > 0 ) == ( ( onlyA & bit ) != 0 ) ? dir : -dir;
        // otherwise the line goes along the side before the perturbation
    }
    return {};
}

/// the direction from p to q for two points that coincide before the perturbation, one of them a crossing X of two cuts and the other
/// one on a cut of X: q - X = t * cross( nt, n ) along the cut by the plane with normal n, and the other plane of X, with normal m,
/// has m.( q - X ) = t * dot( m, cross( nt, n ) ), which is t * d along the cut by tri and -t * d along the one by tri2 (see crossingExact),
/// so the side of q of the other plane gives the sign of t; nullopt if the rule does not apply
std::optional<Vector2i> CutTrianglePredicates::directionAlongCut_( VertId p, VertId q ) const
{
    const auto & points = impl_->points;
    if ( !isCrossing_( p ) )
    {
        if ( !isCrossing_( q ) )
            return {};
        if ( const auto dir = directionAlongCut_( q, p ) )
            return -*dir;
        return {};
    }
    const auto & x = points[p];
    const auto & y = points[q];
    for ( int k = 0; k < 2; ++k )
    {
        const auto & cut = k == 0 ? x.def.tri : x.def.tri2;
        if ( y.def.isVertex() ? !hasVert( cut, y.def.seg[0].id ) : !onPlane( cut, y ) )
            continue;
        const int side = sideOfPlane_( k == 0 ? x.def.tri2 : x.def.tri, q );
        if ( side == 0 )
            return {}; // q is on both cuts
        // orient3d is true where m.( q - X ) < 0
        const int t = ( k == 0 ? -side : side ) * impl_->crossings[x.crossing].dSign;
        const auto dir = lineDirection_( { .plane = cut } );
        if ( dir == Vector2i() )
            return {}; // the planes coincide also after the perturbation, which needs repeated ids
        return t > 0 ? dir : -dir;
    }
    return {};
}

std::optional<Vector2i> CutTrianglePredicates::perturbedDirection_( VertId p, VertId q, bool alike ) const
{
    const bool swapped = q < p;
    const auto key = swapped ? ( std::uint64_t( q ) << 32 ) | std::uint64_t( p ) : ( std::uint64_t( p ) << 32 ) | std::uint64_t( q );
    Vector2i dir;
    if ( !impl_->dirs.if_contains( key, [&] ( const auto & x ) { dir = x.second; } ) )
    {
        std::optional<Vector2i> res;
        if ( alike )
            res = swapped ? shiftDirection_( q, p ) : shiftDirection_( p, q ); // the perturbation moves both points alike
        else
        {
            // a side or a cut of a crossing of two cuts gives the direction faster than the expansion of the perturbation, if it applies
            res = swapped ? directionFromSide_( q, p ) : directionFromSide_( p, q );
            if ( !res )
                res = swapped ? directionAlongCut_( q, p ) : directionAlongCut_( p, q );
            if ( !res )
                res = swapped ? directionByPerturbation( q, p ) : directionByPerturbation( p, q );
        }
        dir = res.value_or( Vector2i() );
        impl_->dirs.emplace( key, dir );
        impl_->cached.store( true, std::memory_order_relaxed );
    }
    if ( dir == Vector2i() )
        return {}; // the points coincide after the perturbation too
    return swapped ? -dir : dir;
}

std::optional<Vector2i> CutTrianglePredicates::directionByPerturbation( VertId p, VertId q ) const
{
    return sosDirection( impl_->points[p].def, impl_->points[q].def, tri_, kx_, ky_ );
}

/// the signs of the projected coordinates of q - p for points p and q defined alike: of the difference of their shifts, with the same scale;
/// nullopt if they have no shifts, or equal ones
std::optional<Vector2i> CutTrianglePredicates::shiftDirection_( VertId p, VertId q ) const
{
    const auto sp = shift_( p ), sq = shift_( q );
    if ( !sp || !sq )
        return {};
    const Vector2i dir( ( sq->x - sp->x ).sign(), ( sq->y - sp->y ).sign() );
    if ( dir == Vector2i() )
        return {};
    return dir;
}

/// the shift of crossing p of an edge of another mesh with the triangle, by which the predicates tell apart the points defined alike:
/// the edge ab moves by delta * u, u = w0 + w1 - a - b toward its wedge w0, w1, by delta infinitely smaller than every term of the perturbation
/// of the vertices, and so its crossing with the plane of normal n by delta * ( u - ( n.u ) / ( n.d ) * d ), d = b - a, along the cut of the plane
/// by the plane of ab and u, into the angle between the cuts by the triangles of the edge (the same for either direction of the segment and
/// either orientation of the triangle); or by -delta * u if the triangles of another edge defined alike are in that angle, so that the sheets
/// of the edges that touch along the segment give contours that do not cross; nullopt without a wedge, or if the segment is parallel to the plane
std::optional<CutTrianglePredicates::Shift> CutTrianglePredicates::shift_( VertId p ) const
{
    const auto & points = impl_->points;
    const auto & pt = points[p];
    const auto & def = pt.def;
    // the segment of a crossing of a side is that side, and a crossing of two cuts has no segment
    if ( def.isVertex() || def.isCrossing() || pt.sides != 0 || !hasWedge( def ) )
        return {};
    const Vector3i64 a( def.seg[0].pt ), b( def.seg[1].pt );
    const auto d = b - a;
    const auto u = Vector3i64( def.wedge[0].pt ) + Vector3i64( def.wedge[1].pt ) - a - b;
    // with the components of the normal below 2^63 (see normalOf), those of d below 2^31 and those of u below 2^32,
    // n.d and n.u are below 3 * 2^95, and the projected coordinates below 6 * 2^126
    const FastInt128 nd = normalDot( normal_, d ), nu = normalDot( normal_, u );
    if ( nd == 0 )
        return {}; // the segment is in the plane before the perturbation, or the triangle has a zero normal
    auto coord = [&] ( int k ) { return Int128Mul256( nd ) * Int128Mul256( u[k] ) - Int128Mul256( nu ) * Int128Mul256( d[k] ); };
    Shift res{ coord( kx_ ), coord( ky_ ), nd < 0 ? -nd : nd };
    // away from the wedge if the triangles of another point defined alike are in its angle; found only here, since points defined alike are rare
    bool away = false;
    for ( auto x = points.beginId(); x < points.endId() && !away; ++x )
        away = x != p && points[x].sides == 0 && hasWedge( points[x].def ) && definedAlike( def, points[x].def ) && nestedIn( def, points[x].def );
    if ( ( nd < 0 ) != away )
    {
        res.x = -res.x;
        res.y = -res.y;
    }
    impl_->shifted.store( true, std::memory_order_relaxed );
    return res;
}

std::optional<CutTrianglePredicates::Line> CutTrianglePredicates::commonLine_( VertId p, VertId q, bool withVertex ) const
{
    const auto & a = impl_->points[p];
    const auto & b = impl_->points[q];
    if ( const int common = a.sides & b.sides )
        return Line{ .side = firstSide( common ) }; // one bit, unless p and q are the same vertex
    if ( a.def.isVertex() || b.def.isVertex() )
    {
        if ( !withVertex )
            return {};
        // a vertex v and a crossing of edge cd with this triangle are on plane vcd; a vertex v and a crossing of a side with a triangle
        // of v are on the plane of that triangle (selfBoolean cuts along it from v, where the triangles share v)
        const auto & v = ( a.def.isVertex() ? a : b ).def.seg[0];
        const auto & x = a.def.isVertex() ? b : a;
        if ( x.def.isCrossing() )
        {
            // a vertex is on a cut of a crossing of two cuts if the plane of the cut has it, as at selfBoolean's folds
            for ( const auto * cut : { &x.def.tri, &x.def.tri2 } )
                if ( hasVert( *cut, v.id ) )
                    return Line{ .plane = *cut };
            return {};
        }
        if ( x.def.isVertex() || ( x.sides != 0 && !hasVert( x.def.tri, v.id ) ) || x.def.seg[0].id == v.id || x.def.seg[1].id == v.id )
            return {};
        return Line{ .plane = x.sides != 0 ? x.def.tri : std::array<PreciseVertCoords, 3>{ v, x.def.seg[0], x.def.seg[1] } };
    }
    // a crossing of two cuts is on both of them, with the points on their planes
    if ( a.def.isCrossing() || b.def.isCrossing() )
    {
        const auto & x = a.def.isCrossing() ? a : b;
        const auto & y = a.def.isCrossing() ? b : a;
        for ( const auto * cut : { &x.def.tri, &x.def.tri2 } )
            if ( onPlane( *cut, y ) )
                return Line{ .plane = *cut };
        return {};
    }
    // a crossing of a side with another triangle is on the plane of that triangle, and so is a crossing of its edge with this triangle
    if ( a.sides != 0 )
        return onPlane( a.def.tri, b ) ? std::optional( Line{ .plane = a.def.tri } ) : std::nullopt;
    if ( b.sides != 0 )
        return onPlane( b.def.tri, a ) ? std::optional( Line{ .plane = b.def.tri } ) : std::nullopt;
    // two crossings of edges of other triangles with this one, where the edges have a common vertex, are on the plane through their three vertices
    const auto & s = a.def.seg;
    const auto & t = b.def.seg;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 2; ++j )
            if ( s[i].id == t[j].id && s[1 - i].id != t[1 - j].id )
                return Line{ .plane = { s[1 - i], s[i], t[1 - j] } };
    return {};
}

Vector2i CutTrianglePredicates::lineDirection_( const Line & l ) const
{
    // the signs of the projected coordinates of the direction: from vertex i to vertex i+1 along side i,
    // and along cross( normal of the triangle, normal of the plane ) on a plane
    if ( l.side >= 0 )
        return sideDirs_[l.side];
    const auto [lx, ly] = lineDirectionOf( normal_, l.plane, kx_, ky_ );
    if ( const Vector2i res( sgn( lx ), sgn( ly ) ); res != Vector2i() )
        return res;
    // only the perturbation gives the direction if the planes are parallel before it, or one of the normals is zero
    return sosCrossDirection( tri_, l.plane, kx_, ky_ );
}

bool CutTrianglePredicates::along_( const Vector2i & dir, VertId p, VertId q ) const
{
    // q - p goes along the line, and the sweep meets p first iff it goes in the direction of the line
    if ( dir.x == 0 && dir.y == 0 )
    {
        fail_(); // the planes coincide also after the perturbation, which needs repeated ids
        return false;
    }
    return less( p, q ) == sweepForward( dir );
}

/// 1 or -1 as point x is on the side of the plane where orient3d is true or false, or 0 if it is on the plane, so on the line
int CutTrianglePredicates::sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const
{
    const auto & p = impl_->points[x].def;
    auto orient = [&] ( const PreciseVertCoords & v )
    {
        return orient3d( { plane[0], plane[1], plane[2], v } );
    };
    auto side = [] ( bool positive ) { return positive ? 1 : -1; };
    if ( p.isCrossing() )
        return crossingSideOfPlane_( plane, x );
    if ( p.isVertex() )
        return hasVert( plane, p.seg[0].id ) ? 0 : side( orient( p.seg[0] ) ); // a vertex of the plane is on it
    // an end of the segment in the plane: the segment meets the plane only there, so the point is on the side of the other end
    const bool in0 = hasVert( plane, p.seg[0].id );
    const bool in1 = hasVert( plane, p.seg[1].id );
    if ( in0 && in1 )
        return 0; // the segment is in the plane
    if ( in0 )
        return side( orient( p.seg[1] ) );
    if ( in1 )
        return side( orient( p.seg[0] ) );
    // the point is on the segment, so on the side of both ends if they are on one side
    const bool o0 = orient( p.seg[0] );
    if ( o0 == orient( p.seg[1] ) )
        return side( o0 );
    if ( sameTriangle( p.tri, plane ) )
        return 0; // the point is on the plane
    // the segment crosses the plane, so the point is on the side of the end it meets first if it meets its triangle before the plane;
    // the boolean's predicate takes any plane through three vertices with distinct ids, as the made-up planes vcd and dce of commonLine_ are,
    // that shares at most two of them with the triangle and none with the segment, as the checks above and in addPoint ensure
    const bool res = segmentIntersectionTriPlaneOrder( { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2], plane[0], plane[1], plane[2] } ) == o0;
#ifndef NDEBUG
    // if the segment crosses the triangle of the plane too, the order along the segment is the one of the boolean's cut, see directionFromSide_
    if ( doTriangleSegmentIntersect( { plane[0], plane[1], plane[2], p.seg[0], p.seg[1] } ) )
        assert( res == ( segmentIntersectionOrder( { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2], plane[0], plane[1], plane[2] } ) == o0 ) );
#endif
    return side( res );
}

/// sideOfPlane_ for a crossing X = n / d of two cuts (see crossingExact) and the plane R with normal m: the sign of m.( X - R0 ) is that
/// of M * d for M = dot( m, n ) - dot( m, R0 ) * d (see planeValue), with the leading coefficients of the perturbed n and d at the limit
/// of cuts parallel before the perturbation (see addCrossing_); or of the perturbed polynomials if M is zero: for the planes of four triangles
/// through one point (common in CAD meshes), for a plane through the limit (as the other triangles in the plane of both cuts are), and for
/// the limit in a triangle with a zero normal, whose parts are zero
int CutTrianglePredicates::crossingSideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const
{
    const auto & pt = impl_->points[x];
    if ( sameTriangle( plane, pt.def.tri ) || sameTriangle( plane, pt.def.tri2 ) )
        return 0; // on the cut by the plane
    const auto & c = impl_->crossings[pt.crossing];
    const auto m = normalOf( plane );
    int s = planeValue( c.parts, m, normalDot( m, Vector3i64( plane[0].pt ) ) ).sign();
    if ( s == 0 )
        s = sosCrossingSide( pt.def, tri_, plane );
    if ( s == 0 )
    {
        fail_(); // on the plane also after the perturbation, which needs repeated ids
        return 1;
    }
    // orient3d is true where m.( X - R0 ) < 0
    return s * c.dSign < 0 ? 1 : -1;
}

int CutTrianglePredicates::exactOrient_( VertId a, VertId b, VertId c ) const
{
    // the determinant of the rows ( nx, ny, d ) of the exact projected points has the sign of the doubled area of abc,
    // since every d is positive; with numerators below 1.3 * 2^126 and denominators below 1.3 * 2^96 (see exactCoords),
    // every 2x2 minor is below 3.4 * 2^252 and fits in 256 bits, and the determinant is below 2^352
    if ( isCrossing_( a ) || isCrossing_( b ) || isCrossing_( c ) )
    {
        const auto & points = impl_->points;
        const auto & crossings = impl_->crossings;
        return orientWide( wideOf( points[a], crossings ), wideOf( points[b], crossings ), wideOf( points[c], crossings ) );
    }
    const auto & p = impl_->points[a].exact;
    const auto & q = impl_->points[b].exact;
    const auto & r = impl_->points[c].exact;
    using M = Int128Mul256;
    const auto det = ( M( q.ny ) * M( r.d ) - M( q.d ) * M( r.ny ) ) * p.nx
                   - ( M( q.nx ) * M( r.d ) - M( q.d ) * M( r.nx ) ) * p.ny
                   + ( M( q.nx ) * M( r.ny ) - M( q.ny ) * M( r.nx ) ) * p.d;
    return det.sign();
}

bool CutTrianglePredicates::ccw( VertId a, VertId b, VertId c ) const
{
    if ( const auto res = approxCcw( approx_[a], approx_[b], approx_[c] ) )
        return *res;
    return ccwExact_( a, b, c );
}

bool CutTrianglePredicates::ccwExact_( VertId a, VertId b, VertId c ) const
{
    // the exact positions are on one line or nearly so: a side through two of the points decides at once
    if ( const auto res = ccwBySides_( a, b, c ) )
        return *res;
    // the other rules are slower, while the sweep asks about the same triangles often:
    // the cache keeps their answers for the triangles with the vertices sorted by ids
    bool flipped = false;
    const UnorientedTriangle key( { a, b, c }, &flipped );
    int s = 0; // 1 or -1 for the sorted vertices, 0 if not found yet
    {
        std::lock_guard lock( impl_->orientsMutex );
        if ( const auto it = impl_->orients.find( key ); it != impl_->orients.end() )
            s = it->second;
    }
    if ( s == 0 )
    {
        s = orientSlow_( key[0], key[1], key[2] );
        std::lock_guard lock( impl_->orientsMutex );
        impl_->orients.emplace( key, s );
        impl_->cached.store( true, std::memory_order_relaxed );
    }
    return ( s > 0 ) != flipped;
}

int CutTrianglePredicates::orientSlow_( VertId a, VertId b, VertId c ) const
{
    // the lines known from the definitions of the points decide even the points on one line before the perturbation
    if ( const auto res = ccwByPlanes_( a, b, c ) )
        return *res ? 1 : -1;
    if ( const int s = exactOrient_( a, b, c ) )
        return s;
    // the points are on one line before the perturbation, and no two of them on a known one: the expansion of the perturbation decides,
    // unless two of them are defined alike, which it moves alike, and which only their shifts tell apart
    if ( const auto res = ccwByShifts_( a, b, c, nullptr ) )
        return *res ? 1 : -1;
    const auto & points = impl_->points;
    if ( const int s = sosOrient( points[a].def, points[b].def, points[c].def, tri_, kx_, ky_ ) )
        return s;
    fail_(); // the points are on one line after the perturbation too, or the expansion reaches a stage whose degrees do not fit
    return -1;
}

std::optional<bool> CutTrianglePredicates::ccwByPerturbation( VertId a, VertId b, VertId c ) const
{
    const auto & points = impl_->points;
    if ( const int s = sosOrient( points[a].def, points[b].def, points[c].def, tri_, kx_, ky_ ) )
        return s > 0;
    return {};
}

std::optional<bool> CutTrianglePredicates::ccwBySides_( VertId a, VertId b, VertId c ) const
{
    const auto & points = impl_->points;
    if ( const int common = points[a].sides & points[b].sides & points[c].sides )
    {
        // three points on one side: they bulge inward a little (infinitely less than the perturbation of the vertices),
        // so that every point inside the side is a reflex vertex of the boundary, where a triangulation adds no sliver triangle
        const auto dir = lineDirection_( { .side = firstSide( common ) } );
        return int( along_( dir, a, b ) ) + int( along_( dir, b, c ) ) + int( along_( dir, c, a ) ) == 1;
    }
    // rotating the vertices keeps the orientation of the triangle, so any two of them on one side can come first,
    // and the triangle is on the left of its sides
    const std::array<VertId, 3> v{ a, b, c };
    for ( int i = 0; i < 3; ++i )
    {
        const VertId p = v[i], q = v[( i + 1 ) % 3];
        if ( const int common = points[p].sides & points[q].sides )
            return along_( lineDirection_( { .side = firstSide( common ) } ), p, q );
    }
    return {};
}

std::optional<bool> CutTrianglePredicates::ccwByLines( VertId a, VertId b, VertId c ) const
{
    // a side decides at once, and a plane needs the side of the third point, so the sides come first
    if ( const auto res = ccwBySides_( a, b, c ) )
        return res;
    return ccwByPlanes_( a, b, c );
}

std::optional<bool> CutTrianglePredicates::ccwByPlanes_( VertId a, VertId b, VertId c ) const
{
    // the lines through a vertex and a crossing come only in the second round: they are needed only at folds, and taking them first was slower
    const auto & points = impl_->points;
    const std::array<VertId, 3> v{ a, b, c };
    for ( int i = 0; i < 6; ++i )
    {
        const VertId p = v[i % 3], q = v[( i + 1 ) % 3], r = v[( i + 2 ) % 3];
        const bool withVertex = i >= 3;
        if ( withVertex && !points[p].def.isVertex() && !points[q].def.isVertex() )
            continue; // the first round found no line through these two
        if ( const auto l = commonLine_( p, q, withVertex ) )
        {
            assert( l->side < 0 ); // no two of the points on one side, see ccwBySides_
            // the left of the line in its direction is the side of the plane opposite to the plane's normal, which is where orient3d is true
            if ( const int side = sideOfPlane_( l->plane, r ) )
                return along_( lineDirection_( *l ), p, q ) == ( side > 0 );
            // r is on the line too: the points of a cut with a crossing of two cuts among them, as the pieces of a cut split at its crossings
            // have, bulge to the left of the line in the direction that the sweep goes along it, as the points of a side bulge inward
            // (see ccwBySides_), which does not depend on the plane that the line is found by (as a made-up plane through the vertices
            // of a triangle in another order), and in valid input, two of the points are defined alike otherwise; but two points defined alike,
            // which only the shifts of their edges move apart, make the doubled area delta * cross( w2 - w1, x - P ) + delta * bulge * ...
            // with the third point x (see ccwByShifts_), so the first-order term of their shifts decides before the bulge, though the shifts
            // are infinitely smaller than the bulge, which decides the triangles with at most one point defined alike
            if ( const auto res = ccwByShifts_( a, b, c, &*l ) )
                return res;
            if ( isCrossing_( a ) || isCrossing_( b ) || isCrossing_( c ) )
                return int( less( a, b ) ) + int( less( b, c ) ) + int( less( c, a ) ) == 1; // along the sweep
            fail_(); // three points on one line also after the perturbation
            return false;
        }
    }
    return {}; // no two of the points are on a known line
}

/// the orientation of triangle abc with two points defined alike, by the shifts of their edges (see shift_), or nullopt if no two of them are;
/// line is the known line of all three points that ccwByPlanes_ found, or null; the failure if the shifts do not tell it either, or if
/// the first-order term of the third point is zero also after the perturbation, and a crossing of two cuts is on the line, among the points
/// or not: then the points of the line bulge (see ccwByPlanes_), so the bulge decides before the second-order term for the triangles of the two
/// points defined alike with that crossing, while neither the order along the line, as for the other points of the line, nor the second-order term,
/// as for the triangles of a third point without a crossing, keeps the axioms of CC systems in every set; that needs the difference of the shifts
/// along the line also after the perturbation, which turns the line, so valid input hardly gives it; without a known line, the second-order term
/// decides then
std::optional<bool> CutTrianglePredicates::ccwByShifts_( VertId a, VertId b, VertId c, const Line * line ) const
{
    const auto & points = impl_->points;
    // rotating the vertices keeps the orientation of the triangle, so the two defined alike can come first
    const std::array<VertId, 3> v{ a, b, c };
    int i = 0;
    while ( i < 3 && !definedAlike( points[v[i]].def, points[v[( i + 1 ) % 3]].def ) )
        ++i;
    if ( i == 3 )
        return {};
    const VertId p = v[i], q = v[( i + 1 ) % 3], r = v[( i + 2 ) % 3];
    const auto sp = shift_( p ), sq = shift_( q );
    if ( !sp || !sq )
    {
        fail_(); // nothing tells apart p and q
        return false;
    }
    // p and q at the common point P are shifted by delta * wp and delta * wq, and r by delta * wr, so the doubled area of the triangle is
    // delta * cross( wq - wp, r - P ) + delta^2 * cross( wq - wp, wr - wp ), where r - P is perturbed, and all its terms come before delta
    const FastInt256 dx = sq->x - sp->x, dy = sq->y - sp->y;
    if ( !definedAlike( points[p].def, points[r].def ) )
    {
        // r - P = ( f.nx * e.d - e.nx * f.d, f.ny * e.d - e.ny * f.d ) / ( e.d * f.d ) with positive denominators, and the numerators
        // below 2 * 1.3 * 2^126 * 1.3 * 2^96 < 2^224 (see exactCoords), or 2 * 2^224 * 2^97 for a crossing r of two cuts (see WideCoords),
        // times the coordinates of the difference of the shifts below 2^130, in the wide integers, since points defined alike are rare
        const auto e = wideOf( points[p], impl_->crossings ), f = wideOf( points[r], impl_->crossings );
        const auto ex = f.nx * e.d - e.nx * f.d, ey = f.ny * e.d - e.ny * f.d;
        if ( const int s = ( dx * ey - dy * ex ).sign() )
            return s > 0;
        // r - P is parallel to the difference of the shifts before the perturbation, so the expansion of the perturbation decides
        auto constant = [] ( const FastInt256 & x ) { return x.sign() ? SosPoly256( std::vector<SosPoly256::Term>{ { 0, x } } ) : SosPoly256(); };
        const SosPoly256 wx = constant( dx ), wy = constant( dy );
        const int sos = sosFirstStage<2>( { &points[p].def, &points[r].def }, tri_, kx_, ky_,
            [&] ( const auto & pts, std::int64_t from, std::int64_t & deg ) -> std::optional<int>
        {
            const SosFraction & pp = *pts[0];
            const SosFraction & rr = *pts[1];
            const SosPoly512 nx = mulAs<FastInt256>( rr.nx, pp.d ) - mulAs<FastInt256>( pp.nx, rr.d );
            const SosPoly512 ny = mulAs<FastInt256>( rr.ny, pp.d ) - mulAs<FastInt256>( pp.ny, rr.d );
            const SignedProduct<SosPoly256, SosPoly512> products[2] = { { &wx, &ny, false }, { &wy, &nx, true } };
            const auto [leadDeg, sign] = lowestTermOfSumOfProducts<FastInt256, FastInt512>( products, 2, from );
            if ( sign == 0 )
                return {};
            deg = leadDeg;
            return sign;
        } ).value_or( 0 );
        if ( sos )
            return sos > 0;
        // a crossing of two cuts is on the line iff the line is a cut of it, as sideOfPlane_ finds it, also for a made-up plane;
        // the scan is rare, as this path is
        auto onLine = [&] ( const CutPoint & x )
        {
            return x.crossing >= 0 && ( sameTriangle( line->plane, x.def.tri ) || sameTriangle( line->plane, x.def.tri2 ) );
        };
        if ( line && std::any_of( points.begin(), points.end(), onLine ) )
        {
            fail_(); // the bulge would decide before the second-order term for the triangles of p and q with that crossing
            return false;
        }
    }
    // the second-order term, with the shift of r of its own scale (zero for a point without a shift):
    // wr - wp = ( Wr * scale - Wp * rScale ) / ( scale * rScale ), the products below 2^129 * 2^96
    FastInt256 rx, ry;
    FastInt128 rScale = 1;
    if ( const auto sr = shift_( r ) )
    {
        rx = sr->x;
        ry = sr->y;
        rScale = sr->scale;
    }
    const FastInt256 fx( rx * sp->scale - sp->x * rScale ), fy( ry * sp->scale - sp->y * rScale );
    if ( const int s = ( dx * fy - dy * fx ).sign() )
        return s > 0;
    fail_(); // the shifts are on one line too
    return false;
}

bool CutTrianglePredicates::ccwFromBehind( VertId b, VertId pivot ) const
{
    // the triangle ( pivot - R * ( 1, eta ), b, pivot ) for the sweep direction ( 1, eta ) has the doubled area
    // R * ( eta * d.x - d.y ) for d = b - pivot, so d.y decides, and only an exactly zero d.y leaves d.x to decide, since eps << eta
    if ( const int c = compareCoord( approx_[b].y, approx_[pivot].y, [&] { return compareExact_( b, pivot, true ); } ) )
        return std::abs( c ) == 1 ? c < 0 : c > 0; // below pivot, or level with it and ahead
    // the points coincide before the perturbation: the direction of the perturbed (or shifted) points, by which less orders them, gives d
    if ( const auto d = coincidentDirection_( pivot, b ) )
        return d->y != 0 ? d->y < 0 : d->x > 0;
    return false; // the failure: points defined alike, not told apart by their shifts
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} // namespace MR
