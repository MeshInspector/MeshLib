#include "MRCutTrianglePredicates.h"
#include "MRFastInt.h"
#include "MRInt64Mul128.h"
#include "MRPrecisePredicates3Poly.h"
#include "MRSparsePolynomial.h"
#include "MRTriMath.h"
#include "MRUnorientedTriangle.h"
#include "MRVarBigInt.h"
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
using detail::cMaxPerturbedPoints;
using detail::cMaxPolyDAll;
using detail::orient3dPoly;
using detail::orient3dVolume;
using detail::PerturbationStages;
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

/// the direction of the line where the planes with normals a and b cross, with components below 2 * 2^126 = 2^127 for the components
/// of the normals below 2^63
Vector3<FastInt128> lineDirectionOf( const Vector3i64 & a, const Vector3i64 & b )
{
    return cross( Vector3i64mul( a ), Vector3i64mul( b ) );
}

bool hasVert( const std::array<PreciseVertCoords, 3> & t, VertId v )
{
    return t[0].id == v || t[1].id == v || t[2].id == v;
}

bool sameTriangle( const std::array<PreciseVertCoords, 3> & a, const std::array<PreciseVertCoords, 3> & b )
{
    return hasVert( b, a[0].id ) && hasVert( b, a[1].id ) && hasVert( b, a[2].id );
}

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
//   (see sosPoint) have the weights of a and b at most 13 and those of the vertices of the triangle at most 9.
// For three points of five vertices each, the most that a predicate here takes, the degrees stay below ( 14^2 * 10^3 )^3 < 2^53,
// since merging two vertices into one never makes the product of their radices larger.
// The vertices are perturbed in stages (see PerturbationStages), so the polynomials stay small, and the expansion stops
// at the first stage where the polynomial of the predicate is not zero.
// The coefficients of a volume are below 2^96 (sums of products of three coordinate differences below 2^31),
// those of nx, ny, d below 2^129, so 128 bits hold the former, 256 bits the latter, 512 bits their products of two
// and 768 bits those of three.

constexpr int cMaxSosVerts = 15; // all vertices of three points
static_assert( cMaxSosVerts <= cMaxPerturbedPoints );

template <typename C>
using SosPolyOf = SparsePolynomial<C, std::int64_t, cMaxPolyDAll>;
using SosPoly256 = SosPolyOf<FastInt256>;
using SosPoly512 = SosPolyOf<FastInt512>;

/// the vertices of the points of one predicate with the bounds of their weights, point after point: seg[0] of a vertex point,
/// or seg[0], seg[1], tri[0], tri[1], tri[2] of a crossing
struct SosVerts
{
    std::array<PreciseVertCoords, cMaxSosVerts> verts;
    std::array<std::int64_t, cMaxSosVerts> bounds; // of the weights of the vertices in the terms of a point
    int num = 0;

    /// adds vertex v, whose weight in a term is at most bound
    void add( const PreciseVertCoords & v, std::int64_t bound )
    {
        assert( num < cMaxSosVerts );
        verts[num] = v;
        bounds[num] = bound;
        ++num;
    }

    /// adds the vertices of point p, and returns the index of the first of them
    int add( const PreciseCutPoint & p )
    {
        const int first = num;
        if ( p.isVertex() )
            add( p.seg[0], 9 );
        else
        {
            add( p.seg[0], 13 );
            add( p.seg[1], 13 );
            for ( const auto & t : p.tri )
                add( t, 9 );
        }
        return first;
    }

    /// the stages of the perturbation of the vertices added: a vertex in several points has the sum of their bounds
    PerturbationStages stages() const
    {
        std::array<std::int64_t, cMaxSosVerts> radices;
        for ( int i = 0; i < num; ++i )
        {
            radices[i] = 1;
            for ( int j = 0; j < num; ++j )
                if ( verts[j].id == verts[i].id )
                    radices[i] += bounds[j];
        }
        return PerturbationStages( verts.data(), radices.data(), num );
    }
};

/// whether point p with its vertices from index first of the stages moves at stage s, having a vertex perturbed first at this stage
bool startsAt( const PerturbationStages & ps, const PreciseCutPoint & p, int first, int s )
{
    for ( int i = first; i < first + ( p.isVertex() ? 1 : 5 ); ++i )
        if ( ps.startsAt( i, s ) )
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

/// the polynomial with the coefficients converted into a wider type
template <typename C, typename P>
SosPolyOf<C> widen( const P & p )
{
    std::vector<typename SosPolyOf<C>::Term> terms;
    terms.reserve( p.get().size() );
    for ( const auto & [deg, c] : p.get() )
        terms.emplace_back( deg, C( c ) );
    return SosPolyOf<C>( std::move( terms ) );
}

// the perturbed vectors of the directions of segments and lines, which only the rare points with zero exact directions need,
// with the coefficients of any size; Vector3 cannot hold polynomials, which have no division
using SosVec = std::array<SosPolyOf<VarBigInt>, 3>;

SosVec sosCoords( const PointDegree & v )
{
    return { sosCoord<VarBigInt>( v, 0 ), sosCoord<VarBigInt>( v, 1 ), sosCoord<VarBigInt>( v, 2 ) };
}

SosVec sub( const SosVec & a, const SosVec & b )
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

SosVec cross( const SosVec & a, const SosVec & b )
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

/// the perturbed normal of the triangle of the vertices from index i of the stages, at stage s
SosVec sosNormal( const PerturbationStages & ps, int i, int s )
{
    const auto a = sosCoords( ps.at( i, s ) );
    return cross( sub( sosCoords( ps.at( i + 1, s ) ), a ), sub( sosCoords( ps.at( i + 2, s ) ), a ) );
}

/// the lowest-degree term of the perturbed volume of the vertices a, b, c, d of the stages (see orient3dPoly) whose exact volume is zero,
/// at the first stage where it is not zero; nullopt if it stays zero, which needs repeated ids
std::optional<VolumePoly<cMaxPolyDAll>::Term> sosLeadOfVolume( const PerturbationStages & ps, int a, int b, int c, int d )
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

/// the perturbed projected point p with its vertices from index first of the stages, at stage s; nullopt if its denominator is zero
/// at this stage, so that only the vertices perturbed at a later stage move the ends of its segment off the plane
std::optional<SosFraction> sosPoint( const PreciseCutPoint & p, const PerturbationStages & ps, int first, int s, int kx, int ky )
{
    SosFraction res;
    if ( p.isVertex() )
    {
        const auto v = ps.at( first, s );
        res.nx = widen<FastInt256>( sosCoord<FastInt128>( v, kx ) );
        res.ny = widen<FastInt256>( sosCoord<FastInt128>( v, ky ) );
        res.d = SosPoly256( std::vector<SosPoly256::Term>{ { 0, FastInt256( 1 ) } } );
        return res;
    }
    const auto a = ps.at( first, s ), b = ps.at( first + 1, s );
    const auto t0 = ps.at( first + 2, s ), t1 = ps.at( first + 3, s ), t2 = ps.at( first + 4, s );
    const auto oa = orient3dPoly<cMaxPolyDAll>( t0, t1, t2, a ), ob = orient3dPoly<cMaxPolyDAll>( t0, t1, t2, b );
    // ( oa * b - ob * a ) / ( oa - ob ), as in exactCoords; the coefficients of the volumes are below 2^96 and those of the coordinates 2^30,
    // and a coefficient of a product sums at most two products of them
    res.nx = mulAs<Int128Mul256>( oa, sosCoord<FastInt128>( b, kx ) ) - mulAs<Int128Mul256>( ob, sosCoord<FastInt128>( a, kx ) );
    res.ny = mulAs<Int128Mul256>( oa, sosCoord<FastInt128>( b, ky ) ) - mulAs<Int128Mul256>( ob, sosCoord<FastInt128>( a, ky ) );
    res.d = widen<FastInt256>( oa - ob );
    if ( res.d.empty() )
        return {};
    if ( res.d.get().front().second.sign() < 0 )
    {
        res.nx = SosPoly256() - res.nx;
        res.ny = SosPoly256() - res.ny;
        res.d = SosPoly256() - res.d;
    }
    return res;
}

/// the lowest-degree term of a polynomial: its degree and the sign of its coefficient, which is 0 for the zero polynomial
struct LeadTerm
{
    std::int64_t deg = 0;
    int sign = 0;
};

LeadTerm leadTerm( const std::pair<std::int64_t, int> & t )
{
    return { t.first, t.second };
}

template <typename C>
LeadTerm leadOf( const SosPolyOf<C> & p )
{
    if ( p.empty() )
        return {};
    return { p.get().front().first, p.get().front().second.sign() };
}

/// the lowest-degree terms of the coordinates of perturbed vector vec( s ) at the first stage s where one of the coordinates with a bit
/// in mask is not zero; a coordinate that is still zero has a leading term of a later stage, so of a larger degree;
/// all signs are zero if the vector stays zero
template <typename F>
std::array<LeadTerm, 3> sosLeadTerms( const PerturbationStages & ps, int mask, const F & vec )
{
    return ps.firstStage( [&] ( int s ) -> std::optional<std::array<LeadTerm, 3>>
    {
        const auto u = vec( s );
        const std::array<LeadTerm, 3> res{ leadOf( u[0] ), leadOf( u[1] ), leadOf( u[2] ) };
        for ( int i = 0; i < 3; ++i )
            if ( ( mask & ( 1 << i ) ) && res[i].sign != 0 )
                return res;
        return {};
    } ).value_or( std::array<LeadTerm, 3>{} );
}

/// the first value that f( x, from ) gives for the perturbed projected points x[i] of pts[i] at the stages of their perturbation
/// where all of them are defined, or the empty value if it gives none; the products of the terms below degree from cancel one another
/// in the polynomial of f, which it found zero at the previous stage if from is not zero (see `from` in lowestTermOfSumOfProducts)
template <int N, typename F>
auto sosFirstStage( const std::array<const PreciseCutPoint *, N> & pts, int kx, int ky, const F & f )
{
    SosVerts r;
    std::array<int, N> first;
    for ( int i = 0; i < N; ++i )
        first[i] = r.add( *pts[i] );
    const auto ps = r.stages();
    std::array<std::optional<SosFraction>, N> x;
    bool prevZero = false; // whether all points were defined at the previous stage, and the polynomial of f zero
    return ps.firstStage( [&] ( int s )
    {
        bool defined = true;
        for ( int i = 0; i < N; ++i )
        {
            // a point keeps its coordinates of the previous stage unless it has a vertex perturbed first at this one
            if ( s == 1 || startsAt( ps, *pts[i], first[i], s ) )
                x[i] = sosPoint( *pts[i], ps, first[i], s, kx, ky );
            defined = defined && x[i].has_value();
        }
        decltype( f( x, std::int64_t{} ) ) res;
        if ( defined ) // otherwise the segment of a point is still in the plane of its triangle
            res = f( x, prevZero ? ps.place( s ) : 0 );
        prevZero = defined && !res;
        return res;
    } );
}

/// the signs of the projected coordinates of a perturbed vector with these leading terms, as the sweep along kx + eta * ky
/// with eps << eta sees them: a coordinate whose leading term has a larger degree than that of the other one is taken as zero
Vector2i sweepSigns( const LeadTerm & x, const LeadTerm & y )
{
    return { y.sign == 0 || x.deg <= y.deg ? x.sign : 0, x.sign == 0 || y.deg <= x.deg ? y.sign : 0 };
}

/// whether the sweep meets the points along a direction with these signs of the projected coordinates in the order
/// of the direction: the first coordinate decides, or the second if the first is exactly zero, since eps << eta
bool sweepForward( const Vector2i & dir )
{
    return dir.x != 0 ? dir.x > 0 : dir.y > 0;
}

/// the signs of the projected coordinates of perturbed q - p, as the sweep along kx + eta * ky with eps << eta sees them
/// (see sweepSigns); nullopt if the points coincide after the perturbation too
std::optional<Vector2i> sosDirection( const PreciseCutPoint & p, const PreciseCutPoint & q, int kx, int ky )
{
    return sosFirstStage<2>( { &p, &q }, kx, ky, [] ( const auto & x, std::int64_t from ) -> std::optional<Vector2i>
    {
        const SosFraction & a = *x[0];
        const SosFraction & b = *x[1];
        // the signs and the leading degrees of the differences of the coordinates of q and p, whose common denominator is positive
        const auto dx = leadTerm( lowestTermOfProductsDiff<FastInt256>( b.nx, a.d, a.nx, b.d, from ) );
        const auto dy = leadTerm( lowestTermOfProductsDiff<FastInt256>( b.ny, a.d, a.ny, b.d, from ) );
        if ( dx.sign == 0 && dy.sign == 0 )
            return {}; // the vertices perturbed so far move both points alike
        // a difference that is still zero has a leading term of a later stage, so of a larger degree
        return sweepSigns( dx, dy );
    } );
}

/// the sign of the doubled area of perturbed triangle pqr in the projection: the determinant of the rows ( nx, ny, d )
/// of the perturbed points, whose denominators d are positive; 0 if the points are on one line after the perturbation too
int sosOrient( const PreciseCutPoint & p, const PreciseCutPoint & q, const PreciseCutPoint & r, int kx, int ky )
{
    return sosFirstStage<3>( { &p, &q, &r }, kx, ky, [] ( const auto & x, std::int64_t from ) -> std::optional<int>
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
        if ( const int sign = lowestTermOfSumOfProducts<FastInt256, FastInt512>( products, 3, from ).second )
            return sign;
        return {}; // the vertices perturbed so far keep the points on one line
    } ).value_or( 0 );
}

/// the signs of the projected coordinates of the perturbed cross( normal of f, normal of g ), as the sweep along kx + eta * ky
/// with eps << eta sees them (see sweepSigns)
Vector2i sosCrossDirection( const std::array<PreciseVertCoords, 3> & f, const std::array<PreciseVertCoords, 3> & g, int kx, int ky )
{
    // a normal has the weight of a vertex at most 9 + 3 = 12
    SosVerts r;
    for ( const auto & v : f )
        r.add( v, 12 );
    for ( const auto & v : g )
        r.add( v, 12 );
    const auto ps = r.stages();
    const auto t = sosLeadTerms( ps, ( 1 << kx ) | ( 1 << ky ), [&] ( int s ) { return cross( sosNormal( ps, 0, s ), sosNormal( ps, 3, s ) ); } );
    return sweepSigns( t[kx], t[ky] );
}

/// the same for the perturbed b - a
Vector2i sosSegmentDirection( const PreciseVertCoords & a, const PreciseVertCoords & b, int kx, int ky )
{
    SosVerts r;
    r.add( a, 9 );
    r.add( b, 9 );
    const auto ps = r.stages();
    const auto t = sosLeadTerms( ps, ( 1 << kx ) | ( 1 << ky ), [&] ( int s ) { return sub( sosCoords( ps.at( 1, s ) ), sosCoords( ps.at( 0, s ) ) ); } );
    return sweepSigns( t[kx], t[ky] );
}

/// a point with what the predicates take from its definition most often
struct CutPoint
{
    PreciseCutPoint def;
    // for an edge-triangle point: the volumes of its triangle with the ends of its segment; if both are zero,
    // the leading coefficients of the perturbed volumes, with zero for the one of the larger degree
    FastInt128 oa{}, ob{};
    int sides = 0; // bit i is set if the point is on side i of the triangle, from vertex i to vertex i+1
};

/// the exact projected coordinates of a point: ( nx / d, ny / d ) with d > 0
struct ExactCoords
{
    FastInt128 nx{}, ny{}, d{};
};

ExactCoords exactCoords( const CutPoint & pt, int kx, int ky )
{
    if ( pt.def.isVertex() )
        return { pt.def.seg[0].pt[kx], pt.def.seg[0].pt[ky], 1 };
    // ( oa * b - ob * a ) / ( oa - ob ) for segment ab (see addPoint), where oa and ob have different signs (or one of them is zero);
    // by Hadamard's inequality |oa|, |ob| < 3^1.5 * 2^93 < 0.65 * 2^96 for coordinate differences below 2^31 (and the leading
    // coefficients of perturbed volumes are smaller), so the numerators are below ( |oa| + |ob| ) * 2^30 < 1.3 * 2^126,
    // and the denominator below 1.3 * 2^96
    const Vector3i & a = pt.def.seg[0].pt;
    const Vector3i & b = pt.def.seg[1].pt;
    ExactCoords res{
        pt.oa * FastInt128( b[kx] ) - pt.ob * FastInt128( a[kx] ),
        pt.oa * FastInt128( b[ky] ) - pt.ob * FastInt128( a[ky] ),
        pt.oa - pt.ob };
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

    /// the directions from p to q of the points that coincide before the perturbation, found from a side and a plane line
    /// or by expanding the perturbation, which is slow, while a sweep compares the same points often;
    /// the key is ( p << 32 ) | q for p < q, and a zero direction marks points that coincide after the perturbation too;
    /// a sweep sorts the points in parallel, hence the map with a mutex in every submap
    phmap::parallel_flat_hash_map_m<std::uint64_t, Vector2i> dirs;

    /// the orientations of the triangles with the vertices sorted by ids that the slower rules decide: 1 if counterclockwise,
    /// -1 otherwise; the sweep asks about the same triangles often, but not in parallel, so one mutex is enough
    HashMap<UnorientedTriangle, int> orients;
    std::mutex orientsMutex;

    /// whether the maps above may have entries, since clearing the empty submaps of the parallel map takes a lock each
    std::atomic<bool> cached{ false };

    std::atomic<bool> failed{ false };

    /// forgets the orders and orientations found by the slower rules
    void clearCaches()
    {
        if ( !cached.exchange( false, std::memory_order_relaxed ) )
            return;
        dirs.clear();
        orients.clear();
    }
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
    tri_ = tri;
    approx_.clear();
    impl_->points.clear();
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
        // the vertices are on one line before the perturbation: drop the axis of the leading term of the perturbed normal
        SosVerts r;
        for ( const auto & v : tri )
            r.add( v, 12 ); // a normal has the weight of a vertex at most 9 + 3
        const auto ps = r.stages();
        const auto n = sosLeadTerms( ps, 7, [&] ( int s ) { return sosNormal( ps, 0, s ); } );
        LeadTerm lead;
        for ( int i = 0; i < 3; ++i )
            if ( n[i].sign != 0 && ( lead.sign == 0 || n[i].deg < lead.deg ) )
            {
                dropAx = i;
                lead = n[i];
            }
        assert( lead.sign != 0 ); // the perturbed points of distinct ids are not on one line
        dropSign = lead.sign;
    }
    kx_ = ( dropAx + 1 ) % 3;
    ky_ = ( dropAx + 2 ) % 3;
    if ( dropSign < 0 )
        std::swap( kx_, ky_ );
}

void CutTrianglePredicates::addPoint( VertId v, const PreciseCutPoint & p )
{
    auto & points = impl_->points;
    if ( points.size() <= v )
    {
        points.resize( v + 1 );
        approx_.resize( v + 1 );
    }
    auto & pt = points[v];
    if ( pt.def.seg[0].id.valid() )
        impl_->clearCaches(); // the point is set again: forget the orders found for its old definition
    pt.def = p;
    auto corner = [&] ( const PreciseVertCoords & x )
    {
        for ( int i = 0; i < 3; ++i )
            if ( tri_[i].id == x.id )
                return i;
        return -1;
    };
    if ( p.isVertex() )
    {
        const int i = corner( p.seg[0] );
        if ( i < 0 )
            fail_(); // not a vertex of the triangle, so on none of its sides, while commonLine_ takes the lines through a vertex
        pt.sides = i >= 0 ? ( 1 << i ) | ( 1 << ( ( i + 2 ) % 3 ) ) : 0;
        pt.oa = pt.ob = 0;
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        return;
    }
    const int i = corner( p.seg[0] ), j = corner( p.seg[1] );
    pt.sides = i >= 0 && j >= 0 ? 1 << ( j == ( i + 1 ) % 3 ? i : j ) : 0;
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
        pt.oa = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
        pt.ob = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
        if ( pt.oa == 0 && pt.ob == 0 )
        {
            // the segment is in the plane, and only the perturbation moves its ends off it: as eps -> 0, the crossing tends to
            // the point where the leading terms of the perturbed volumes cancel, or to the end whose volume has the leading term
            // of the larger degree, since the other volume is infinitely larger then
            SosVerts r;
            r.add( p );
            const auto ps = r.stages();
            const auto va = sosLeadOfVolume( ps, 2, 3, 4, 0 );
            const auto vb = sosLeadOfVolume( ps, 2, 3, 4, 1 );
            assert( va && vb ); // the perturbed volumes of distinct vertices are not zero
            const auto & [degA, leadA] = *va;
            const auto & [degB, leadB] = *vb;
            // the leading terms decide as orient3d does, which ties the degrees above to its perturbation
            assert( ( leadA > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } ) );
            assert( ( leadB > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } ) );
            pt.oa = degA <= degB ? leadA : 0;
            pt.ob = degB <= degA ? leadB : 0;
            crosses = ( leadA > 0 ) != ( leadB > 0 );
        }
        else
        {
            // an end exactly on the plane is on the side the perturbation puts it on
            const bool sa = pt.oa != 0 ? pt.oa > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } );
            const bool sb = pt.ob != 0 ? pt.ob > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } );
            crosses = sa != sb;
        }
    }
    if ( !crosses )
    {
        // repeated ids, a segment off the sides with another triangle, or the segment does not cross the plane, also after the perturbation
        fail_();
        pt.oa = 1; // any point of the segment, since the answers are arbitrary now
        pt.ob = -1;
    }
    // that the crossing is inside its triangle stays the caller's precondition, checked in Debug only, since it takes several orient3d
    assert( !crosses || doTriangleSegmentIntersect( { p.tri[0], p.tri[1], p.tri[2], p.seg[0], p.seg[1] } ) );
    // the point is a + s * ( b - a ), s = oa / ( oa - ob ), where oa and ob have different signs (or one of them is zero)
    const Vector3d a( p.seg[0].pt ), b( p.seg[1].pt );
    const double da = toDouble( pt.oa );
    const double s = da / ( da - toDouble( pt.ob ) );
    approx_[v] = { a[kx_] + s * ( b[kx_] - a[kx_] ), a[ky_] + s * ( b[ky_] - a[ky_] ) };
}

/// the order of points p and q along the projected coordinate y if yFirst (x otherwise), whose rounded values cannot tell it,
/// and along the other coordinate where the points are level along the first one: -1 or 1 as p comes before or after q along
/// the first coordinate, -2 or 2 along the second one, 0 if the points coincide before the perturbation
int CutTrianglePredicates::compareExact_( VertId p, VertId q, bool yFirst ) const
{
    const auto a = exactCoords( impl_->points[p], kx_, ky_ );
    const auto b = exactCoords( impl_->points[q], kx_, ky_ );
    if ( const int c = compareExactCoord( a, b, yFirst ) )
        return c;
    // the exact coordinates found above decide the second one where its rounded values cannot
    const int k = yFirst ? 0 : 1;
    return 2 * compareCoord( approx_[p][k], approx_[q][k], [&] { return compareExactCoord( a, b, !yFirst ); } );
}

Vector2i CutTrianglePredicates::segmentDirection_( const PreciseVertCoords & a, const PreciseVertCoords & b ) const
{
    // only the perturbation gives the direction if the segment goes along the dropped axis, as a side of a triangle
    // with a zero exact normal can
    const auto u = Vector3i64( b.pt ) - Vector3i64( a.pt );
    const Vector2i res( sgn( u[kx_] ), sgn( u[ky_] ) );
    if ( res.x != 0 || res.y != 0 )
        return res;
    return sosSegmentDirection( a, b, kx_, ky_ );
}

bool CutTrianglePredicates::less( VertId p, VertId q ) const
{
    if ( p == q )
        return false;
    // along x, then along y
    if ( const int c = compareCoord( approx_[p].x, approx_[q].x, [&] { return compareExact_( p, q, false ); } ) )
        return c < 0;
    return lessCoincident_( p, q );
}

bool CutTrianglePredicates::lessCoincident_( VertId p, VertId q ) const
{
    const auto & da = impl_->points[p].def;
    const auto & db = impl_->points[q].def;
    // along a segment s0 -> s1, the order of the perturbed points is that of the sweep if the sweep meets s0 first
    auto alongSweep = [&] ( const PreciseVertCoords & s0, const PreciseVertCoords & s1 )
    {
        return sweepForward( segmentDirection_( s0, s1 ) );
    };
    if ( !da.isVertex() && !db.isVertex() )
    {
        const bool same = da.seg[0].id == db.seg[0].id && da.seg[1].id == db.seg[1].id;
        const bool opposite = da.seg[0].id == db.seg[1].id && da.seg[1].id == db.seg[0].id;
        if ( same || opposite )
        {
            // two crossings of one side with other triangles: in the order the boolean's cut puts them along it
            if ( sameTriangle( da.tri, db.tri ) )
            {
                fail_(); // the same point twice
                return p < q;
            }
            const bool aFirst = segmentIntersectionOrder( { da.seg[0], da.seg[1], da.tri[0], da.tri[1], da.tri[2], db.tri[0], db.tri[1], db.tri[2] } );
            return aFirst == alongSweep( da.seg[0], da.seg[1] );
        }
    }
    // a crossing of a side and an end of the side: the crossing is strictly inside the side
    if ( !da.isVertex() && db.isVertex() )
    {
        if ( db.seg[0].id == da.seg[0].id )
            return !alongSweep( da.seg[0], da.seg[1] );
        if ( db.seg[0].id == da.seg[1].id )
            return !alongSweep( da.seg[1], da.seg[0] );
    }
    if ( da.isVertex() && !db.isVertex() )
    {
        if ( da.seg[0].id == db.seg[0].id )
            return alongSweep( db.seg[0], db.seg[1] );
        if ( da.seg[0].id == db.seg[1].id )
            return alongSweep( db.seg[1], db.seg[0] );
    }
    if ( const auto dir = perturbedDirection_( p, q ) )
        return sweepForward( *dir );
    fail_(); // the points coincide after the perturbation too
    return p < q;
}

std::optional<Vector2i> CutTrianglePredicates::directionFromSide_( VertId p, VertId q ) const
{
    // a point on a side and a point not on it, both on one plane line: the latter is strictly on the left of the side,
    // so the direction from the former to it is the direction of the line, signed to point to the left of the side
    const auto & a = impl_->points[p];
    const auto & b = impl_->points[q];
    const int onlyA = a.sides & ~b.sides;
    const int onlyB = b.sides & ~a.sides;
    if ( onlyA == 0 && onlyB == 0 )
        return {};
    // no line through a vertex: a vertex v exactly at a crossing of segment cd is on the line of cd, so plane vcd has a zero normal,
    // and exactly at a crossing of a side with a triangle of v it is on that side, so this triangle is flat: no line has a direction then
    const auto l = commonLine_( p, q, false );
    if ( !l || l->side >= 0 )
        return {};
    const auto line = lineDirectionOf( normal_, normalOf( l->plane ) );
    const FastInt128 lx = line[kx_], ly = line[ky_];
    if ( lx == 0 && ly == 0 )
        return {}; // the line has no exact direction: the planes are parallel before the perturbation, or a normal is zero
    const Vector2i dir( sgn( lx ), sgn( ly ) );
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

std::optional<Vector2i> CutTrianglePredicates::perturbedDirection_( VertId p, VertId q ) const
{
    const bool swapped = q < p;
    const auto key = swapped ? ( std::uint64_t( q ) << 32 ) | std::uint64_t( p ) : ( std::uint64_t( p ) << 32 ) | std::uint64_t( q );
    Vector2i dir;
    if ( !impl_->dirs.if_contains( key, [&] ( const auto & x ) { dir = x.second; } ) )
    {
        // the line through a point on a side gives the direction faster than the expansion of the perturbation, if it applies
        auto res = swapped ? directionFromSide_( q, p ) : directionFromSide_( p, q );
        if ( !res )
            res = swapped ? directionByPerturbation( q, p ) : directionByPerturbation( p, q );
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
    return sosDirection( impl_->points[p].def, impl_->points[q].def, kx_, ky_ );
}

std::optional<CutTrianglePredicates::Line> CutTrianglePredicates::commonLine_( VertId p, VertId q, bool withVertex ) const
{
    const auto & a = impl_->points[p];
    const auto & b = impl_->points[q];
    if ( const int common = a.sides & b.sides )
        return Line{ .side = std::countr_zero( unsigned( common ) ) }; // one bit, unless p and q are the same vertex
    if ( a.def.isVertex() || b.def.isVertex() )
    {
        if ( !withVertex )
            return {};
        // a vertex v and a crossing of edge cd with this triangle are on plane vcd; a vertex v and a crossing of a side with a triangle
        // of v are on the plane of that triangle (selfBoolean cuts along it from v, where the triangles share v)
        const auto & v = ( a.def.isVertex() ? a : b ).def.seg[0];
        const auto & x = a.def.isVertex() ? b : a;
        if ( x.def.isVertex() || ( x.sides != 0 && !hasVert( x.def.tri, v.id ) ) || x.def.seg[0].id == v.id || x.def.seg[1].id == v.id )
            return {};
        return Line{ .plane = x.sides != 0 ? x.def.tri : std::array<PreciseVertCoords, 3>{ v, x.def.seg[0], x.def.seg[1] } };
    }
    // a crossing of a side with another triangle is on the plane of that triangle, and so is a crossing of its edge with this triangle
    auto onPlane = [] ( const std::array<PreciseVertCoords, 3> & t, const CutPoint & x )
    {
        return x.sides != 0 ? sameTriangle( t, x.def.tri ) : hasVert( t, x.def.seg[0].id ) && hasVert( t, x.def.seg[1].id );
    };
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
        return segmentDirection_( tri_[l.side], tri_[( l.side + 1 ) % 3] );
    const auto line = lineDirectionOf( normal_, normalOf( l.plane ) );
    const Vector2i res( sgn( line[kx_] ), sgn( line[ky_] ) );
    if ( res.x != 0 || res.y != 0 )
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

bool CutTrianglePredicates::sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const
{
    const auto & p = impl_->points[x].def;
    auto orient = [&] ( const PreciseVertCoords & v )
    {
        return orient3d( { plane[0], plane[1], plane[2], v } );
    };
    if ( p.isVertex() )
    {
        if ( hasVert( plane, p.seg[0].id ) )
        {
            fail_(); // a vertex of the plane, so on the line
            return false;
        }
        return orient( p.seg[0] );
    }
    // an end of the segment in the plane: the segment meets the plane only there, so the point is on the side of the other end
    const bool in0 = hasVert( plane, p.seg[0].id );
    const bool in1 = hasVert( plane, p.seg[1].id );
    if ( in0 && in1 )
    {
        fail_(); // the segment is in the plane, so the point is on the line
        return false;
    }
    if ( in0 )
        return orient( p.seg[1] );
    if ( in1 )
        return orient( p.seg[0] );
    // the point is on the segment, so on the side of both ends if they are on one side
    const bool o0 = orient( p.seg[0] );
    if ( o0 == orient( p.seg[1] ) )
        return o0;
    if ( sameTriangle( p.tri, plane ) )
    {
        fail_(); // the point is on the plane, so on the line
        return false;
    }
    // the segment crosses the plane, so the point is on the side of the end it meets first if it meets its triangle before the plane;
    // the boolean's predicate takes any plane through three vertices with distinct ids, as the made-up planes vcd and dce of commonLine_ are,
    // that shares at most two of them with the triangle and none with the segment, as the checks above and in addPoint ensure
    const bool res = segmentIntersectionTriPlaneOrder( { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2], plane[0], plane[1], plane[2] } ) == o0;
#ifndef NDEBUG
    // if the segment crosses the triangle of the plane too, the order along the segment is the one of the boolean's cut, see lessCoincident_
    if ( doTriangleSegmentIntersect( { plane[0], plane[1], plane[2], p.seg[0], p.seg[1] } ) )
        assert( res == ( segmentIntersectionOrder( { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2], plane[0], plane[1], plane[2] } ) == o0 ) );
#endif
    return res;
}

int CutTrianglePredicates::exactOrient_( VertId a, VertId b, VertId c ) const
{
    // the determinant of the rows ( nx, ny, d ) of the exact projected points has the sign of the doubled area of abc,
    // since every d is positive; with numerators below 1.3 * 2^126 and denominators below 1.3 * 2^96 (see exactCoords),
    // every 2x2 minor is below 3.4 * 2^252 and fits in 256 bits, and the determinant is below 2^352
    const auto p = exactCoords( impl_->points[a], kx_, ky_ );
    const auto q = exactCoords( impl_->points[b], kx_, ky_ );
    const auto r = exactCoords( impl_->points[c], kx_, ky_ );
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
    // the points are on one line before the perturbation, and no two of them on a known one: the expansion of the perturbation decides
    const auto & points = impl_->points;
    if ( const int s = sosOrient( points[a].def, points[b].def, points[c].def, kx_, ky_ ) )
        return s;
    fail_(); // the points are on one line after the perturbation too
    return -1;
}

std::optional<bool> CutTrianglePredicates::ccwByPerturbation( VertId a, VertId b, VertId c ) const
{
    const auto & points = impl_->points;
    if ( const int s = sosOrient( points[a].def, points[b].def, points[c].def, kx_, ky_ ) )
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
        const auto dir = lineDirection_( Line{ .side = std::countr_zero( unsigned( common ) ) } );
        return int( along_( dir, a, b ) ) + int( along_( dir, b, c ) ) + int( along_( dir, c, a ) ) == 1;
    }
    // rotating the vertices keeps the orientation of the triangle, so any two of them on one side can come first,
    // and the triangle is on the left of its sides
    const std::array<VertId, 3> v{ a, b, c };
    for ( int i = 0; i < 3; ++i )
    {
        const VertId p = v[i], q = v[( i + 1 ) % 3];
        if ( const int common = points[p].sides & points[q].sides )
            return along_( lineDirection_( Line{ .side = std::countr_zero( unsigned( common ) ) } ), p, q );
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
            return along_( lineDirection_( *l ), p, q ) == sideOfPlane_( l->plane, r );
        }
    }
    return {}; // no two of the points are on a known line
}

bool CutTrianglePredicates::ccwFromBehind( VertId b, VertId pivot ) const
{
    // the triangle ( pivot - R * ( 1, eta ), b, pivot ) for the sweep direction ( 1, eta ) has the doubled area
    // R * ( eta * d.x - d.y ) for d = b - pivot, so d.y decides, and only an exactly zero d.y leaves d.x to decide, since eps << eta
    if ( const int c = compareCoord( approx_[b].y, approx_[pivot].y, [&] { return compareExact_( b, pivot, true ); } ) )
        return std::abs( c ) == 1 ? c < 0 : c > 0; // below pivot, or level with it and ahead
    // the points coincide before the perturbation: d goes along their common side if they have one, in the order less finds there;
    // less orders any other pair by the direction of the perturbed points (see lessCoincident_), which gives d
    Vector2i dir;
    if ( const int common = impl_->points[pivot].sides & impl_->points[b].sides )
    {
        dir = lineDirection_( Line{ .side = std::countr_zero( unsigned( common ) ) } );
        if ( !along_( dir, pivot, b ) )
            dir = -dir;
    }
    else if ( const auto d = perturbedDirection_( pivot, b ) )
        dir = *d;
    else
    {
        fail_(); // the points coincide after the perturbation too
        return false;
    }
    return dir.y != 0 ? dir.y < 0 : dir.x > 0;
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} // namespace MR
