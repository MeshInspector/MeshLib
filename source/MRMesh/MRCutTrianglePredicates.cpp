#include "MRCutTrianglePredicates.h"
#include "MRFastInt.h"
#include "MRInt64Mul128.h"
#include "MRSparsePolynomial.h"
#include "MRVarBigInt.h"
#include "MRphmap.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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

// the rounded projected coordinates of an edge-triangle point are within 2^-19 integer units from the exact ones, as a ratio
// of two volumes converted in double, times a coordinate difference within 2^31; so a larger difference of two of them decides
constexpr double cApproxTol = 0x1p-17;

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

/// the signed volume of tetrahedron abcd, positive iff orient3d( a, b, c, d ), computed as in MRPrecisePredicates3.cpp
FastInt128 volume( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
{
    const Vector3i64 x( a - d );
    const Vector3i64 y( b - d );
    const Vector3i64 z( c - d );
    return Int64Mul128( x.x ) * Int64Mul128( y.y * z.z - y.z * z.y )
         - Int64Mul128( x.y ) * Int64Mul128( y.x * z.z - y.z * z.x )
         + Int64Mul128( x.z ) * Int64Mul128( y.x * z.y - y.y * z.x );
}

/// the normal of a triangle: every coordinate is below 2^30 by absolute value, as the range factor 0.99 of ConvertToIntVector
/// ensures, so every coordinate difference is below 2^31, and every component of the normal is below 2 * 2^62 = 2^63
Vector3i64 normalOf( const std::array<PreciseVertCoords, 3> & t )
{
    const Vector3i64 a( t[0].pt );
    return cross( Vector3i64( t[1].pt ) - a, Vector3i64( t[2].pt ) - a );
}

/// component k of cross( a, b ) for vectors with components below 2^63, as normals have: it is below 2 * 2^126 = 2^127
FastInt128 crossComponent( const Vector3i64 & a, const Vector3i64 & b, int k )
{
    const int i = ( k + 1 ) % 3, j = ( k + 2 ) % 3;
    return Int64Mul128( a[i] ) * Int64Mul128( b[j] ) - Int64Mul128( a[j] ) * Int64Mul128( b[i] );
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
// Coordinate i of the vertex of rank r (the ranks go by ascending ids among the vertices of one predicate) gets eps^( w[i] * P[r] ),
// w = 9, 3, 1 for x, y, z, as in getPointDegrees of MRPrecisePredicates3.cpp, which takes P[r] = 27^r. Here P[0] = 1 and
// P[r+1] = P[r] * ( 1 + B[r] ), where B[r] bounds the weight of the vertex of rank r in every term of the polynomials
// of the predicate (the sum of w[i] over the coordinates of the vertex in the term, with repetitions). As a term has the weights
// W[s] <= B[s], the vertices below rank r add at most sum_{s<r} B[s] * P[s] = P[r] - 1 to its degree, so the degrees order
// the terms by the weights of the vertices from the largest id down, as orient3d's degrees do with weights below 27:
// all predicates take one perturbation, eps^( w[i] * K^r ) for a large enough K, which is the perturbation of the boolean.
// The bounds follow from the definitions of the points:
// - a coordinate of a vertex has the weight of that vertex at most 9;
// - every term of the perturbed volume of t[0] t[1] t[2] x takes one coordinate from each row t[j] - x, all in different columns,
//   so it has the weight of every t[j] at most 9 and that of x at most 9 + 3 + 1 = 13;
// - so the projected coordinates nx = oa * b[k] - ob * a[k] and d = oa - ob of the crossing of segment ab with the plane
//   of a triangle (see sosPoint) have the weights of a and b at most 13 and those of the vertices of the triangle at most 9,
//   and the coordinates of a vertex point at most 9;
// - a product of polynomials of several points (as in a determinant of their coordinates) has the weight of a vertex
//   at most the sum of its bounds in the points that contain it.
// For three points of five vertices each, the most that a predicate here takes, P stays below ( 14^2 * 10^3 )^3 < 2^53,
// since merging two vertices into one never makes the product larger.
// The vertices are perturbed in stages: at stage s only the s smallest ids move, so the polynomials stay small, and
// a polynomial not zero at stage s already has the leading term of the full one, since all terms involving the other
// vertices come after it. If it was zero at stage s - 1, its terms without the vertex of rank s - 1, perturbed last,
// cancel one another, so only the products with the terms of that vertex, of degrees from P[s-1] up, need computing.
// The coefficients of a volume are below 2^96 (sums of products of three coordinate differences below 2^31),
// those of nx, ny, d below 2^129, so 128 bits hold the former, 256 bits the latter, 512 bits their products of two
// and 768 bits those of three.

constexpr int cMaxSosVerts = 15; // all vertices of three points

template <typename C>
using SosPolyOf = SparsePolynomial<C, std::int64_t, std::numeric_limits<std::int64_t>::max()>;
using SosPoly = SosPolyOf<VarBigInt>;
using SosVec = std::array<SosPoly, 3>;

/// the vertices of one predicate by ascending ids, with the bounds of their weights and their place values (B and P above),
/// of which the first `stage` ones are perturbed
struct SosRanks
{
    std::array<VertId, cMaxSosVerts> ids;
    std::array<std::int64_t, cMaxSosVerts> bounds;
    std::array<std::int64_t, cMaxSosVerts> places;
    int n = 0;
    int stage = 0;

    /// adds vertex v, whose weight in a term grows by at most bound
    void add( VertId v, int bound )
    {
        const int i = int( std::lower_bound( ids.begin(), ids.begin() + n, v ) - ids.begin() );
        if ( i < n && ids[i] == v )
        {
            bounds[i] += bound;
            return;
        }
        assert( n < cMaxSosVerts );
        for ( int j = n; j > i; --j )
        {
            ids[j] = ids[j - 1];
            bounds[j] = bounds[j - 1];
        }
        ids[i] = v;
        bounds[i] = bound;
        ++n;
    }

    /// adds the vertices of point p with the bounds of their weights in its perturbed projected coordinates
    void add( const PreciseCutPoint & p )
    {
        if ( p.isVertex() )
        {
            add( p.seg[0].id, 9 );
            return;
        }
        add( p.seg[0].id, 13 );
        add( p.seg[1].id, 13 );
        for ( const auto & t : p.tri )
            add( t.id, 9 );
    }

    /// computes the place values, once all vertices are added
    void setPlaces()
    {
        std::int64_t place = 1;
        for ( int i = 0; i < n; ++i )
        {
            places[i] = place;
            assert( place <= std::numeric_limits<std::int64_t>::max() / ( 1 + bounds[i] ) );
            place *= 1 + bounds[i];
        }
    }

    int rank( VertId v ) const
    {
        const auto it = std::lower_bound( ids.begin(), ids.begin() + n, v );
        assert( it != ids.begin() + n && *it == v );
        return int( it - ids.begin() );
    }
};

int signOf( const VarBigInt & c )
{
    return c.sign();
}

int signOf( FastInt128 c )
{
    return sgn( c );
}

template <int nBits>
int signOf( const FastInt<nBits> & c )
{
    return c.sign();
}

template <typename C = VarBigInt>
SosPolyOf<C> sosConst( std::int64_t c )
{
    SosPolyOf<C> res;
    if ( c != 0 )
        res = SosPolyOf<C>( std::vector<typename SosPolyOf<C>::Term>{ { std::int64_t( 0 ), C( c ) } } );
    return res;
}

template <typename C = VarBigInt>
std::array<SosPolyOf<C>, 3> sosCoords( const PreciseVertCoords & v, const SosRanks & r )
{
    constexpr std::int64_t w[3] = { 9, 3, 1 };
    const int rank = r.rank( v.id );
    std::array<SosPolyOf<C>, 3> res;
    for ( int i = 0; i < 3; ++i )
        res[i] = rank < r.stage
            ? SosPolyOf<C>( C( std::int64_t( v.pt[i] ) ), w[i] * r.places[rank], C( std::int64_t( 1 ) ) )
            : sosConst<C>( v.pt[i] );
    return res;
}

template <typename P>
std::array<P, 3> sub( const std::array<P, 3> & a, const std::array<P, 3> & b )
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

template <typename P>
std::array<P, 3> cross( const std::array<P, 3> & a, const std::array<P, 3> & b )
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

/// the perturbed volume of tetrahedron t[0] t[1] t[2] x, positive iff orient3d( t[0], t[1], t[2], x ), see volume above:
/// the determinant of the rows t[j] - x expanded at once, as orient3dPoly of MRPrecisePredicates3.cpp does it, from the elements
/// delta + eps^( degree of t[j] ) - eps^( degree of x ); 128 bits are enough for its coefficients, products of three coordinate
/// differences below 2^31
template <typename C>
SosPolyOf<C> sosVolume( const std::array<PreciseVertCoords, 3> & t, const PreciseVertCoords & x, const SosRanks & r )
{
    constexpr std::int64_t w[3] = { 9, 3, 1 };
    struct Element
    {
        std::pair<std::int64_t, std::int64_t> terms[3]; // degree and coefficient
        int num = 0;
    };
    const int rx = r.rank( x.id );
    Element m[3][3];
    for ( int row = 0; row < 3; ++row )
    {
        const int rt = r.rank( t[row].id );
        for ( int col = 0; col < 3; ++col )
        {
            auto & e = m[row][col];
            if ( const std::int64_t delta = std::int64_t( t[row].pt[col] ) - x.pt[col] )
                e.terms[e.num++] = { 0, delta };
            if ( rt < r.stage )
                e.terms[e.num++] = { w[col] * r.places[rt], 1 };
            if ( rx < r.stage )
                e.terms[e.num++] = { w[col] * r.places[rx], -1 };
        }
    }
    // the sum over the permutations of the columns of the signed products of three elements, one from every row
    constexpr int perms[6][4] = { { 0, 1, 2, 1 }, { 1, 2, 0, 1 }, { 2, 0, 1, 1 }, { 0, 2, 1, -1 }, { 2, 1, 0, -1 }, { 1, 0, 2, -1 } };
    std::vector<typename SosPolyOf<C>::Term> terms;
    terms.reserve( 6 * 27 );
    for ( const auto & p : perms )
    {
        const Element & e0 = m[0][p[0]];
        const Element & e1 = m[1][p[1]];
        const Element & e2 = m[2][p[2]];
        for ( int k0 = 0; k0 < e0.num; ++k0 )
            for ( int k1 = 0; k1 < e1.num; ++k1 )
            {
                const std::int64_t c01 = p[3] * e0.terms[k0].second * e1.terms[k1].second; // below 2^62
                const std::int64_t d01 = e0.terms[k0].first + e1.terms[k1].first;
                for ( int k2 = 0; k2 < e2.num; ++k2 )
                    terms.emplace_back( d01 + e2.terms[k2].first, C( Int64Mul128( c01 ) * Int64Mul128( e2.terms[k2].second ) ) );
            }
    }
    return SosPolyOf<C>::fromUnsortedTerms( std::move( terms ) );
}

/// the perturbed normal of triangle t
SosVec sosNormal( const std::array<PreciseVertCoords, 3> & t, const SosRanks & r )
{
    const auto a = sosCoords( t[0], r );
    return cross( sub( sosCoords( t[1], r ), a ), sub( sosCoords( t[2], r ), a ) );
}

using SosPoly128 = SosPolyOf<FastInt128>;
using SosPoly256 = SosPolyOf<FastInt256>;
using SosPoly512 = SosPolyOf<FastInt512>;

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

/// a point with perturbed coordinates: ( nx / d, ny / d ) in the projection
struct SosFraction
{
    SosPoly256 nx, ny, d;
};

/// the perturbed projected point, or nullopt if its denominator is zero at this stage, so that only the vertices
/// perturbed at a later stage move the ends of its segment off the plane
std::optional<SosFraction> sosPoint( const PreciseCutPoint & p, const SosRanks & r, int kx, int ky )
{
    SosFraction res;
    if ( p.isVertex() )
    {
        const auto c = sosCoords<FastInt128>( p.seg[0], r );
        res = { widen<FastInt256>( c[kx] ), widen<FastInt256>( c[ky] ), sosConst<FastInt256>( 1 ) };
        return res;
    }
    const auto a = sosCoords<FastInt128>( p.seg[0], r ), b = sosCoords<FastInt128>( p.seg[1], r );
    const auto oa = sosVolume<FastInt128>( p.tri, p.seg[0], r ), ob = sosVolume<FastInt128>( p.tri, p.seg[1], r );
    // ( oa * b - ob * a ) / ( oa - ob ), as in CutTrianglePredicates::exactCoords_; the coefficients of the volumes are below 2^96
    // and those of the coordinates 2^30, and a coefficient of a product sums at most two products of them
    res.nx = mulAs<Int128Mul256>( oa, b[kx] ) - mulAs<Int128Mul256>( ob, a[kx] );
    res.ny = mulAs<Int128Mul256>( oa, b[ky] ) - mulAs<Int128Mul256>( ob, a[ky] );
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

/// the lowest-degree term of a * b - c * d, with the products of the coefficients in 512 bits
LeadTerm leadOfProductsDiff( const SosPoly256 & a, const SosPoly256 & b, const SosPoly256 & c, const SosPoly256 & d )
{
    const auto [deg, sign] = lowestTermOfProductsDiff<FastInt256>( a, b, c, d );
    return { deg, sign };
}

template <typename C>
LeadTerm leadOf( const SosPolyOf<C> & p )
{
    return p.empty() ? LeadTerm{} : LeadTerm{ p.get().front().first, signOf( p.get().front().second ) };
}

/// the lowest-degree terms of the coordinates of a perturbed vector vec(), at the first stage where one of the coordinates
/// with a bit in mask is not zero; a coordinate that is still zero has a leading term of a later stage, so of a larger degree;
/// all signs are zero if the vector stays zero
template <typename F>
std::array<LeadTerm, 3> sosLeadTerms( SosRanks & r, int mask, const F & vec )
{
    for ( r.stage = 1; r.stage <= r.n; ++r.stage )
    {
        const auto u = vec();
        const std::array<LeadTerm, 3> res{ leadOf( u[0] ), leadOf( u[1] ), leadOf( u[2] ) };
        for ( int i = 0; i < 3; ++i )
            if ( ( mask & ( 1 << i ) ) && res[i].sign != 0 )
                return res;
    }
    return {};
}

/// the signs of the projected coordinates of a perturbed vector with these leading terms, as the sweep along kx + eta * ky
/// with eps << eta sees them: a coordinate whose leading term has a larger degree than that of the other one is taken as zero
Vector2i sweepSigns( const LeadTerm & x, const LeadTerm & y )
{
    return { y.sign == 0 || x.deg <= y.deg ? x.sign : 0, x.sign == 0 || y.deg <= x.deg ? y.sign : 0 };
}

Vector2i sweepSigns( const std::array<LeadTerm, 3> & t, int kx, int ky )
{
    return sweepSigns( t[kx], t[ky] );
}

/// whether the sweep meets the points along a direction with these signs of the projected coordinates in the order
/// of the direction: the first coordinate decides, or the second if the first is exactly zero, since eps << eta
bool sweepForward( const Vector2i & dir )
{
    return dir.x != 0 ? dir.x > 0 : dir.y > 0;
}

/// a product of two polynomials, negated if neg, as a term of a sum
template <typename A, typename B>
struct SignedProduct
{
    const A * a = nullptr;
    const B * b = nullptr;
    bool neg = false;
};

/// the lowest-degree term of the sum of the products, found as in lowestTermOfProductsDiff: the terms of every product
/// in the order of increasing degree, and only till the first degree with a coefficient that is not zero
template <typename A, typename B>
LeadTerm leadOfSumOfProducts( const SignedProduct<A, B> * products, int num )
{
    struct RowTerm
    {
        std::int64_t deg;
        int p, i, j; // the product and the indices of the terms in its factors
    };
    auto greater = [] ( const RowTerm & x, const RowTerm & y ) { return x.deg > y.deg; };
    std::vector<RowTerm> heap;
    for ( int p = 0; p < num; ++p )
    {
        const auto & a = products[p].a->get();
        const auto & b = products[p].b->get();
        if ( !b.empty() )
            for ( int i = 0; i < int( a.size() ); ++i )
                heap.push_back( { a[i].first + b[0].first, p, i, 0 } );
    }
    std::make_heap( heap.begin(), heap.end(), greater );
    while ( !heap.empty() )
    {
        const auto deg = heap.front().deg;
        decltype( products->a->get()[0].second * products->b->get()[0].second ) coeff;
        do
        {
            std::pop_heap( heap.begin(), heap.end(), greater );
            auto r = heap.back();
            heap.pop_back();
            const auto & pr = products[r.p];
            const auto & a = pr.a->get();
            const auto & b = pr.b->get();
            const auto prod = a[r.i].second * b[r.j].second;
            if ( pr.neg )
                coeff -= prod;
            else
                coeff += prod;
            if ( ++r.j < int( b.size() ) )
            {
                r.deg = a[r.i].first + b[r.j].first;
                heap.push_back( r );
                std::push_heap( heap.begin(), heap.end(), greater );
            }
        }
        while ( !heap.empty() && heap.front().deg == deg );
        if ( const int sign = coeff.sign() )
            return { deg, sign };
    }
    return {};
}

/// the terms of the coordinates of a point without and with the vertex perturbed last, of degrees below and from its place
/// (the other vertices add less to a degree, see the place values above)
void splitAt( const SosFraction & x, std::int64_t place, SosFraction & low, SosFraction & high )
{
    auto split = [place] ( const SosPoly256 & a, SosPoly256 & l, SosPoly256 & h )
    {
        const auto & t = a.get();
        const auto mid = std::lower_bound( t.begin(), t.end(), place, [] ( const auto & term, std::int64_t d ) { return term.first < d; } );
        l = SosPoly256( std::vector( t.begin(), mid ) );
        h = SosPoly256( std::vector( mid, t.end() ) );
    };
    split( x.nx, low.nx, high.nx );
    split( x.ny, low.ny, high.ny );
    split( x.d, low.d, high.d );
}

/// the signs of the projected coordinates of perturbed q - p, as the sweep along kx + eta * ky with eps << eta sees them
/// (see sweepSigns); nullopt if the points coincide after the perturbation too
std::optional<Vector2i> sosDirection( const PreciseCutPoint & p, const PreciseCutPoint & q, int kx, int ky )
{
    SosRanks r;
    r.add( p );
    r.add( q );
    r.setPlaces();
    bool prevZero = false; // whether both points were defined at the previous stage, and the differences of their coordinates zero
    for ( r.stage = 1; r.stage <= r.n; ++r.stage )
    {
        const auto a = sosPoint( p, r, kx, ky );
        const auto b = sosPoint( q, r, kx, ky );
        if ( !a || !b )
        {
            prevZero = false;
            continue; // the segment of a point is still in the plane of its triangle
        }
        // the signs and the leading degrees of the differences of the coordinates of q and p, whose common denominator is positive
        LeadTerm x, y;
        if ( !prevZero )
        {
            x = leadOfProductsDiff( b->nx, a->d, a->nx, b->d );
            y = leadOfProductsDiff( b->ny, a->d, a->ny, b->d );
        }
        else
        {
            // the terms without the vertex perturbed last make the differences of the previous stage, which are zero, so only
            // the products with that vertex in at least one factor remain, e.g. hb.nx * a.d + lb.nx * ha.d - ha.nx * b.d - la.nx * hb.d,
            // which leaves out the products of the terms of the previous stage that cancel one another
            SosFraction la, ha, lb, hb;
            splitAt( *a, r.places[r.stage - 1], la, ha );
            splitAt( *b, r.places[r.stage - 1], lb, hb );
            using P = SignedProduct<SosPoly256, SosPoly256>;
            const P px[4] = { { &hb.nx, &a->d, false }, { &lb.nx, &ha.d, false }, { &ha.nx, &b->d, true }, { &la.nx, &hb.d, true } };
            const P py[4] = { { &hb.ny, &a->d, false }, { &lb.ny, &ha.d, false }, { &ha.ny, &b->d, true }, { &la.ny, &hb.d, true } };
            x = leadOfSumOfProducts( px, 4 );
            y = leadOfSumOfProducts( py, 4 );
        }
        // a difference that is still zero has a leading term of a later stage, so of a larger degree
        if ( x.sign != 0 || y.sign != 0 )
            return sweepSigns( x, y );
        // otherwise the vertices perturbed so far move both points alike
        prevZero = true;
    }
    return {};
}

/// the sign of the doubled area of perturbed triangle pqr in the projection: the determinant of the rows ( nx, ny, d )
/// of the perturbed points, whose denominators d are positive; 0 if the points are on one line after the perturbation too
int sosOrient( const PreciseCutPoint & p, const PreciseCutPoint & q, const PreciseCutPoint & r, int kx, int ky )
{
    SosRanks ranks;
    ranks.add( p );
    ranks.add( q );
    ranks.add( r );
    ranks.setPlaces();
    // the cofactor expansion of det( h, u, v ) along row h, negated if neg: three products of an element of h and a minor
    std::array<SosPoly512, 9> minors;
    std::array<SignedProduct<SosPoly256, SosPoly512>, 9> products;
    int num = 0;
    auto addCofactors = [&] ( const SosFraction & h, const SosFraction & u, const SosFraction & v, bool neg )
    {
        minors[num] = mulAs<FastInt256>( u.ny, v.d ) - mulAs<FastInt256>( u.d, v.ny );
        products[num] = { &h.nx, &minors[num], neg };
        ++num;
        minors[num] = mulAs<FastInt256>( u.nx, v.d ) - mulAs<FastInt256>( u.d, v.nx );
        products[num] = { &h.ny, &minors[num], !neg };
        ++num;
        minors[num] = mulAs<FastInt256>( u.nx, v.ny ) - mulAs<FastInt256>( u.ny, v.nx );
        products[num] = { &h.d, &minors[num], neg };
        ++num;
    };
    bool prevZero = false; // whether all points were defined at the previous stage, and their determinant zero
    for ( ranks.stage = 1; ranks.stage <= ranks.n; ++ranks.stage )
    {
        const auto a = sosPoint( p, ranks, kx, ky );
        const auto b = sosPoint( q, ranks, kx, ky );
        const auto c = sosPoint( r, ranks, kx, ky );
        if ( !a || !b || !c )
        {
            prevZero = false;
            continue; // the segment of a point is still in the plane of its triangle
        }
        num = 0;
        SosFraction la, ha, lb, hb, lc, hc;
        if ( !prevZero )
            addCofactors( *a, *b, *c, false );
        else
        {
            // the terms without the vertex perturbed last make the determinant of the previous stage, which is zero, so only the terms
            // with that vertex in at least one row remain: det( ha, b, c ) + det( la, hb, c ) + det( la, lb, hc ),
            // which leaves out the products of the terms of the previous stage that cancel one another
            const auto place = ranks.places[ranks.stage - 1];
            splitAt( *a, place, la, ha );
            splitAt( *b, place, lb, hb );
            splitAt( *c, place, lc, hc );
            addCofactors( ha, *b, *c, false );
            addCofactors( hb, la, *c, true );
            addCofactors( hc, la, lb, false );
        }
        const auto lead = leadOfSumOfProducts( products.data(), num );
        if ( lead.sign != 0 )
            return lead.sign;
        // otherwise the vertices perturbed so far keep the points on one line
        prevZero = true;
    }
    return 0;
}

/// the side of perturbed plane t of perturbed crossing p, +1 where orient3d with t is true: the sign of the volume of t with p,
/// ( oa * vb - ob * va ) / ( oa - ob ) for the volumes oa, ob of the triangle of p and va, vb of t with the ends of its segment;
/// 0 if p is on the plane after the perturbation too
int sosSideOfPlane( const std::array<PreciseVertCoords, 3> & t, const PreciseCutPoint & p )
{
    SosRanks r;
    r.add( p );
    for ( const auto & v : t )
        r.add( v.id, 9 ); // as a vertex of the triangle of a volume
    r.setPlaces();
    for ( r.stage = 1; r.stage <= r.n; ++r.stage )
    {
        const auto oa = sosVolume<FastInt128>( p.tri, p.seg[0], r );
        const auto ob = sosVolume<FastInt128>( p.tri, p.seg[1], r );
        const auto d = leadOf( oa - ob );
        if ( d.sign == 0 )
            continue; // the segment is still in the plane of its triangle
        const auto va = sosVolume<FastInt128>( t, p.seg[0], r );
        const auto vb = sosVolume<FastInt128>( t, p.seg[1], r );
        // the volumes are within 2^96, and their products within 2^192
        if ( const int s = signOfProductsDiff<Int128Mul256>( oa, vb, ob, va ) )
            return s * d.sign;
        // otherwise the vertices perturbed so far keep the point on the plane
    }
    return 0;
}

/// the signs of the projected coordinates of the perturbed cross( normal of f, normal of g ), as the sweep along kx + eta * ky
/// with eps << eta sees them: a coordinate whose leading term has a larger degree than that of the other one is taken as zero
Vector2i sosCrossDirection( const std::array<PreciseVertCoords, 3> & f, const std::array<PreciseVertCoords, 3> & g, int kx, int ky )
{
    // a normal has the weight of a vertex at most 9 + 3 = 12
    SosRanks r;
    for ( const auto & v : f )
        r.add( v.id, 12 );
    for ( const auto & v : g )
        r.add( v.id, 12 );
    r.setPlaces();
    return sweepSigns( sosLeadTerms( r, ( 1 << kx ) | ( 1 << ky ), [&] { return cross( sosNormal( f, r ), sosNormal( g, r ) ); } ), kx, ky );
}

/// the same for the perturbed b - a
Vector2i sosSegmentDirection( const PreciseVertCoords & a, const PreciseVertCoords & b, int kx, int ky )
{
    SosRanks r;
    r.add( a.id, 9 );
    r.add( b.id, 9 );
    r.setPlaces();
    return sweepSigns( sosLeadTerms( r, ( 1 << kx ) | ( 1 << ky ), [&] { return sub( sosCoords( b, r ), sosCoords( a, r ) ); } ), kx, ky );
}

} // anonymous namespace

/// the directions from p to q of the points that coincide before the perturbation, found from a side and a plane line
/// or by expanding the perturbation, which is slow, while a sweep compares the same points often;
/// the key is ( p << 32 ) | q for p < q, and a zero direction marks points that coincide after the perturbation too;
/// a sweep sorts the points in parallel, hence the mutex
struct CutTrianglePredicates::Cache
{
    HashMap<std::uint64_t, Vector2i> perturbedDirs;
    std::mutex mutex;
};

CutTrianglePredicates::CutTrianglePredicates() : cache_( std::make_unique<Cache>() )
{
}

CutTrianglePredicates::~CutTrianglePredicates() = default;

void CutTrianglePredicates::reset( const std::array<PreciseVertCoords, 3> & tri )
{
    tri_ = tri;
    approx_.clear();
    points_.clear();
    cache_->perturbedDirs.clear();
    failed_.store( false, std::memory_order_relaxed );

    normal_ = normalOf( tri );
    // drop the axis most aligned with the normal; order the two kept axes so that the projection keeps the orientation around the normal
    int dropAx = 0;
    for ( int i = 1; i < 3; ++i )
        if ( std::abs( normal_[i] ) > std::abs( normal_[dropAx] ) )
            dropAx = i;
    int dropSign = sgn( normal_[dropAx] );
    if ( dropSign == 0 )
    {
        // the vertices are on one line before the perturbation: drop the axis of the leading term of the perturbed normal
        SosRanks r;
        for ( const auto & v : tri )
            r.add( v.id, 12 );
        r.setPlaces();
        const auto n = sosLeadTerms( r, 7, [&] { return sosNormal( tri, r ); } );
        for ( int i = 0; i < 3; ++i )
            if ( n[i].sign != 0 && ( dropSign == 0 || n[i].deg < n[dropAx].deg ) )
            {
                dropAx = i;
                dropSign = n[i].sign;
            }
        if ( dropSign == 0 )
            fail_(); // two vertices with the same id
    }
    kx_ = ( dropAx + 1 ) % 3;
    ky_ = ( dropAx + 2 ) % 3;
    if ( dropSign < 0 )
        std::swap( kx_, ky_ );
}

void CutTrianglePredicates::addPoint( VertId v, const PreciseCutPoint & p )
{
    if ( points_.size() <= v )
    {
        points_.resize( v + 1 );
        approx_.resize( v + 1 );
    }
    auto & pt = points_[v];
    if ( pt.def.seg[0].id.valid() )
        cache_->perturbedDirs.clear(); // the point is set again: forget the orders found for its old definition
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
        pt.sides = i >= 0 ? ( 1 << i ) | ( 1 << ( ( i + 2 ) % 3 ) ) : 0;
        pt.oa = pt.ob = 0;
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        return;
    }
    const int i = corner( p.seg[0] ), j = corner( p.seg[1] );
    pt.sides = i >= 0 && j >= 0 ? 1 << ( j == ( i + 1 ) % 3 ? i : j ) : 0;
    pt.oa = volume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
    pt.ob = volume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
    if ( pt.oa == 0 && pt.ob == 0 )
    {
        // the segment is in the plane, and only the perturbation moves its ends off it: as eps -> 0, the crossing tends to
        // the point where the leading terms of the perturbed volumes cancel, or to the end whose volume has the leading term
        // of the larger degree, since the other volume is infinitely larger then
        SosRanks r;
        r.add( p );
        r.setPlaces();
        r.stage = r.n;
        const auto va = sosVolume<FastInt128>( p.tri, p.seg[0], r );
        const auto vb = sosVolume<FastInt128>( p.tri, p.seg[1], r );
        if ( va.empty() || vb.empty() )
        {
            fail_(); // repeated ids
            pt.oa = 1; // any point, since the answers are arbitrary now
            pt.ob = -1;
        }
        else
        {
            const auto & [degA, leadA] = va.get().front();
            const auto & [degB, leadB] = vb.get().front();
            // the leading terms decide as orient3d does, which ties the degrees above to its perturbation
            assert( ( leadA > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } ) );
            assert( ( leadB > 0 ) == orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } ) );
            pt.oa = degA <= degB ? leadA : 0;
            pt.ob = degB <= degA ? leadB : 0;
            if ( ( leadA > 0 ) == ( leadB > 0 ) )
                fail_(); // the segment does not cross the plane after the perturbation
        }
    }
    else
    {
        // an end exactly on the plane is on the side the perturbation puts it on
        const bool sa = pt.oa != 0 ? pt.oa > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[0] } );
        const bool sb = pt.ob != 0 ? pt.ob > 0 : orient3d( { p.tri[0], p.tri[1], p.tri[2], p.seg[1] } );
        if ( sa == sb )
            fail_(); // the segment does not cross the plane
    }
    // the point is a + s * ( b - a ), s = oa / ( oa - ob ), where oa and ob have different signs (or one of them is zero)
    const double da = toDouble( pt.oa );
    const double s = da / ( da - toDouble( pt.ob ) );
    const Vector3d a( p.seg[0].pt ), b( p.seg[1].pt );
    approx_[v] = { a[kx_] + s * ( b[kx_] - a[kx_] ), a[ky_] + s * ( b[ky_] - a[ky_] ) };
}

CutTrianglePredicates::Coords CutTrianglePredicates::exactCoords_( VertId v ) const
{
    const auto & pt = points_[v];
    if ( pt.def.isVertex() )
        return { pt.def.seg[0].pt[kx_], pt.def.seg[0].pt[ky_], 1 };
    // ( oa * b - ob * a ) / ( oa - ob ) for segment ab (see addPoint), where oa and ob have different signs (or one of them is zero);
    // by Hadamard's inequality |oa|, |ob| < 3^1.5 * 2^93 < 0.65 * 2^96 for coordinate differences below 2^31 (and the leading
    // coefficients of perturbed volumes are smaller), so the numerators are below ( |oa| + |ob| ) * 2^30 < 1.3 * 2^126,
    // and the denominator below 1.3 * 2^96
    const Vector3i & a = pt.def.seg[0].pt;
    const Vector3i & b = pt.def.seg[1].pt;
    Coords res{
        pt.oa * FastInt128( b[kx_] ) - pt.ob * FastInt128( a[kx_] ),
        pt.oa * FastInt128( b[ky_] ) - pt.ob * FastInt128( a[ky_] ),
        pt.oa - pt.ob };
    if ( res.d < 0 )
    {
        res.nx = -res.nx;
        res.ny = -res.ny;
        res.d = -res.d;
    }
    return res;
}

int CutTrianglePredicates::compareExact_( VertId p, VertId q, bool y ) const
{
    // the products of a numerator below 1.3 * 2^126 and a denominator below 1.3 * 2^96 (see exactCoords_) fit in 256 bits
    const auto a = exactCoords_( p );
    const auto b = exactCoords_( q );
    const auto l = Int128Mul256( y ? a.ny : a.nx ) * Int128Mul256( b.d );
    const auto r = Int128Mul256( y ? b.ny : b.nx ) * Int128Mul256( a.d );
    return l < r ? -1 : ( r < l ? 1 : 0 );
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
    const auto & a = approx_[p];
    const auto & b = approx_[q];
    if ( a.x < b.x - cApproxTol )
        return true;
    if ( b.x < a.x - cApproxTol )
        return false;
    if ( const int c = compareExact_( p, q, false ) )
        return c < 0;
    if ( a.y < b.y - cApproxTol )
        return true;
    if ( b.y < a.y - cApproxTol )
        return false;
    if ( const int c = compareExact_( p, q, true ) )
        return c < 0;
    return lessCoincident_( p, q );
}

bool CutTrianglePredicates::lessCoincident_( VertId p, VertId q ) const
{
    const auto & da = points_[p].def;
    const auto & db = points_[q].def;
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
    const auto & a = points_[p];
    const auto & b = points_[q];
    const int onlyA = a.sides & ~b.sides;
    const int onlyB = b.sides & ~a.sides;
    if ( onlyA == 0 && onlyB == 0 )
        return {};
    const auto l = commonLine_( p, q, true );
    if ( !l || l->side >= 0 )
        return {};
    const auto n = normalOf( l->plane );
    const FastInt128 lx = crossComponent( normal_, n, kx_ );
    const FastInt128 ly = crossComponent( normal_, n, ky_ );
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
    {
        std::lock_guard lock( cache_->mutex );
        if ( const auto it = cache_->perturbedDirs.find( key ); it != cache_->perturbedDirs.end() )
        {
            if ( it->second == Vector2i() )
                return {}; // the points coincide after the perturbation too
            return swapped ? -it->second : it->second;
        }
    }
    // the line through a point on a side gives the direction faster than the expansion of the perturbation, if it applies
    auto res = swapped ? directionFromSide_( q, p ) : directionFromSide_( p, q );
    if ( !res )
        res = swapped ? directionByPerturbation( q, p ) : directionByPerturbation( p, q );
    std::lock_guard lock( cache_->mutex );
    cache_->perturbedDirs[key] = res.value_or( Vector2i() );
    if ( !res )
        return {};
    return swapped ? -*res : *res;
}

std::optional<Vector2i> CutTrianglePredicates::directionByPerturbation( VertId p, VertId q ) const
{
    return sosDirection( points_[p].def, points_[q].def, kx_, ky_ );
}

std::optional<CutTrianglePredicates::Line> CutTrianglePredicates::commonLine_( VertId p, VertId q, bool withVertex ) const
{
    const auto & a = points_[p];
    const auto & b = points_[q];
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
    auto onPlane = [] ( const std::array<PreciseVertCoords, 3> & t, const Point & x )
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
    const auto n = normalOf( l.plane );
    const Vector2i res( sgn( crossComponent( normal_, n, kx_ ) ), sgn( crossComponent( normal_, n, ky_ ) ) );
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

bool CutTrianglePredicates::sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, const Point & x ) const
{
    const auto & p = x.def;
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
    const auto va = volume( plane[0].pt, plane[1].pt, plane[2].pt, p.seg[0].pt );
    const auto vb = volume( plane[0].pt, plane[1].pt, plane[2].pt, p.seg[1].pt );
    if ( va != 0 && vb != 0 && ( va > 0 ) == ( vb > 0 ) )
        return va > 0;
    // the point is a + s * ( b - a ) for s = oa / ( oa - ob ) (see addPoint), so its volume with the plane is ( oa * vb - ob * va ) / ( oa - ob ),
    // where oa - ob is not zero and has the sign of oa > ob; all volumes are within 2^96, and their products within 2^192
    const auto num = Int128Mul256( x.oa ) * Int128Mul256( vb ) - Int128Mul256( x.ob ) * Int128Mul256( va );
    if ( const int s = num.sign() )
        return ( s > 0 ) == ( x.oa > x.ob );
    // the point is on the plane before the perturbation: it is on the side of both ends if they are on one side after it
    const bool o0 = va != 0 ? va > 0 : orient( p.seg[0] );
    const bool o1 = vb != 0 ? vb > 0 : orient( p.seg[1] );
    if ( o0 == o1 )
        return o0;
    if ( sameTriangle( p.tri, plane ) )
    {
        fail_(); // the point is on the plane, so on the line
        return false;
    }
    if ( const int s = sosSideOfPlane( plane, p ) )
        return s > 0;
    fail_(); // the point is on the plane after the perturbation too
    return false;
}

int CutTrianglePredicates::exactOrient_( VertId a, VertId b, VertId c ) const
{
    // the determinant of the rows ( nx, ny, d ) of the exact projected points has the sign of the doubled area of abc,
    // since every d is positive; with numerators below 1.3 * 2^126 and denominators below 1.3 * 2^96 (see exactCoords_),
    // every 2x2 minor is below 3.4 * 2^252 and fits in 256 bits, and the determinant is below 2^352
    const auto p = exactCoords_( a );
    const auto q = exactCoords_( b );
    const auto r = exactCoords_( c );
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
    // the exact positions are on one line or nearly so: the lines known from the definitions of the points decide even the former
    if ( const auto res = ccwByLines( a, b, c ) )
        return *res;
    if ( const int s = exactOrient_( a, b, c ) )
        return s > 0;
    // the points are on one line before the perturbation, and no two of them on a known one
    if ( const int s = sosOrient( points_[a].def, points_[b].def, points_[c].def, kx_, ky_ ) )
        return s > 0;
    fail_(); // the points are on one line after the perturbation too
    return false;
}

std::optional<bool> CutTrianglePredicates::ccwByLines( VertId a, VertId b, VertId c ) const
{
    if ( const int common = points_[a].sides & points_[b].sides & points_[c].sides )
    {
        // three points on one side: they bulge inward a little (infinitely less than the perturbation of the vertices),
        // so that every point inside the side is a reflex vertex of the boundary, where a triangulation adds no sliver triangle
        const auto dir = lineDirection_( Line{ .side = std::countr_zero( unsigned( common ) ) } );
        return int( along_( dir, a, b ) ) + int( along_( dir, b, c ) ) + int( along_( dir, c, a ) ) == 1;
    }
    // rotating the vertices keeps the orientation of the triangle, so any two of them on one line can come first;
    // the lines through a vertex and a crossing come only in the second round: they are needed only at folds, and taking them first was slower
    const std::array<VertId, 3> v{ a, b, c };
    for ( int i = 0; i < 6; ++i )
    {
        const VertId p = v[i % 3], q = v[( i + 1 ) % 3], r = v[( i + 2 ) % 3];
        const auto l = commonLine_( p, q, i >= 3 );
        if ( !l )
            continue;
        const bool forward = along_( lineDirection_( *l ), p, q );
        if ( l->side >= 0 )
            return forward; // the triangle is on the left of its sides, and r is not on this one
        // the left of the line in its direction is the side of the plane opposite to the plane's normal, which is where orient3d is true
        return forward == sideOfPlane_( l->plane, points_[r] );
    }
    return {}; // no two of the points are on a known line
}

bool CutTrianglePredicates::ccwFromBehind( VertId b, VertId pivot ) const
{
    // the triangle ( pivot - R * ( 1, eta ), b, pivot ) for the sweep direction ( 1, eta ) has the doubled area
    // R * ( eta * d.x - d.y ) for d = b - pivot, so d.y decides, and only an exactly zero d.y leaves d.x to decide, since eps << eta
    const auto & x = approx_[b];
    const auto & o = approx_[pivot];
    if ( x.y < o.y - cApproxTol )
        return true;
    if ( o.y < x.y - cApproxTol )
        return false;
    if ( const int c = compareExact_( b, pivot, true ) )
        return c < 0;
    if ( o.x < x.x - cApproxTol )
        return true;
    if ( x.x < o.x - cApproxTol )
        return false;
    if ( const int c = compareExact_( b, pivot, false ) )
        return c > 0;
    // the points coincide before the perturbation: d goes along the line through them if one is known from their definitions,
    // otherwise the expansion of the perturbation gives its direction
    Vector2i dir;
    if ( const auto l = commonLine_( pivot, b, true ) )
    {
        dir = lineDirection_( *l );
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
