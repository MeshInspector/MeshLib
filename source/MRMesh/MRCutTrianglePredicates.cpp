#include "MRCutTrianglePredicates.h"
#include "MRInt64Mul128.h"
#include "MRSparsePolynomial.h"
#include "MRVarBigInt.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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

/// the normal of a triangle, with the components within 2^63, since every coordinate difference is within 2^31
Vector3i64 normalOf( const std::array<PreciseVertCoords, 3> & t )
{
    const Vector3i64 a( t[0].pt );
    return cross( Vector3i64( t[1].pt ) - a, Vector3i64( t[2].pt ) - a );
}

/// the sign of component k of cross( a, b ) for vectors with components within 2^63
int crossComponentSign( const Vector3i64 & a, const Vector3i64 & b, int k )
{
    const int i = ( k + 1 ) % 3, j = ( k + 2 ) % 3;
    const auto l = Int64Mul128( a[i] ) * Int64Mul128( b[j] );
    const auto r = Int64Mul128( a[j] ) * Int64Mul128( b[i] );
    return l > r ? 1 : ( l < r ? -1 : 0 );
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
// Every coordinate i of the vertex of rank r (the ranks go by ascending ids among the vertices of one comparison)
// gets eps^( w[i] * 64^r ), w = 9, 3, 1 for x, y, z, as in getPointDegrees of MRPrecisePredicates3.cpp with 64
// in place of 27. No vertex weighs 64 or more in a term below (7 coordinates at most, each of weight 9 at most),
// so the terms are ordered by the weights of the vertices from the largest id down, which is also how orient3d orders
// them: this is the perturbation of the boolean. A degree stores the weight of rank r in bits 6r..6r+5.
// The vertices are perturbed in stages: at stage s only the s smallest ids move, so the polynomials stay small, and
// a polynomial not zero at stage s already has the leading term of the full one, since all terms involving the other
// vertices come after it.

constexpr int cSosBits = 6;
constexpr int cMaxSosStage = 10; // all vertices of two points

using SosPoly = SparsePolynomial<VarBigInt, std::int64_t, std::numeric_limits<std::int64_t>::max()>;
using SosVec = std::array<SosPoly, 3>;

/// the vertices of one comparison, by ascending ids, of which the first `stage` ones are perturbed
struct SosRanks
{
    std::array<VertId, cMaxSosStage> ids;
    int n = 0;
    int stage = 0;

    void add( VertId v )
    {
        if ( std::find( ids.begin(), ids.begin() + n, v ) != ids.begin() + n )
            return;
        assert( n < int( ids.size() ) );
        ids[n++] = v;
    }

    void add( const PreciseVertCoords * vs, int num )
    {
        for ( int i = 0; i < num; ++i )
            add( vs[i].id );
    }

    int rank( VertId v ) const
    {
        const auto it = std::lower_bound( ids.begin(), ids.begin() + n, v );
        assert( it != ids.begin() + n && *it == v );
        return int( it - ids.begin() );
    }
};

SosPoly sosConst( std::int64_t c )
{
    SosPoly res;
    if ( c != 0 )
        res = SosPoly( std::vector<SosPoly::Term>{ { std::int64_t( 0 ), VarBigInt( c ) } } );
    return res;
}

SosVec sosCoords( const PreciseVertCoords & v, const SosRanks & r )
{
    constexpr std::int64_t w[3] = { 9, 3, 1 };
    const int rank = r.rank( v.id );
    SosVec res;
    for ( int i = 0; i < 3; ++i )
        res[i] = rank < r.stage
            ? SosPoly( VarBigInt( std::int64_t( v.pt[i] ) ), w[i] << ( cSosBits * rank ), VarBigInt( std::int64_t( 1 ) ) )
            : sosConst( v.pt[i] );
    return res;
}

SosVec sub( const SosVec & a, const SosVec & b )
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

SosVec cross( const SosVec & a, const SosVec & b )
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

SosPoly dot( const SosVec & a, const SosVec & b )
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// a point as the perturbation needs it: a vertex, or a segment crossing the plane of a triangle
std::array<PreciseVertCoords, 5> sosVerts( const PreciseCutPoint & p )
{
    return { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2] };
}

int sosNumVerts( const PreciseCutPoint & p )
{
    return p.isVertex() ? 1 : 5;
}

/// a point with perturbed coordinates: ( nx / d, ny / d ) in the projection
struct SosFraction
{
    SosPoly nx, ny, d;
};

/// the perturbed projected point, or nullopt if it is not defined before the perturbation
std::optional<SosFraction> sosPoint( const PreciseCutPoint & p, const SosRanks & r, int kx, int ky )
{
    SosFraction res;
    if ( p.isVertex() )
    {
        auto c = sosCoords( p.seg[0], r );
        res = { std::move( c[kx] ), std::move( c[ky] ), sosConst( 1 ) };
        return res;
    }
    const auto a = sosCoords( p.seg[0], r ), b = sosCoords( p.seg[1], r );
    const auto t0 = sosCoords( p.tri[0], r ), t1 = sosCoords( p.tri[1], r ), t2 = sosCoords( p.tri[2], r );
    auto volume = [&] ( const SosVec & x ) { return dot( sub( t0, x ), cross( sub( t1, x ), sub( t2, x ) ) ); };
    const auto oa = volume( a ), ob = volume( b );
    // ( oa * b - ob * a ) / ( oa - ob ), as in CutTrianglePredicates::exactCoord_
    res = { oa * b[kx] - ob * a[kx], oa * b[ky] - ob * a[ky], oa - ob };
    if ( res.d.empty() || res.d.get().front().first != 0 )
        return {}; // zero denominator without the perturbation
    if ( res.d.get().front().second.sign() < 0 )
    {
        res.nx = SosPoly() - res.nx;
        res.ny = SosPoly() - res.ny;
        res.d = SosPoly() - res.d;
    }
    return res;
}

/// the lowest-degree term of a polynomial: its degree and the sign of its coefficient, which is 0 for the zero polynomial
struct LeadTerm
{
    std::int64_t deg = 0;
    int sign = 0;
};

/// the lowest-degree term of a * b - c * d, found as signOfProductsDiff in MRSparsePolynomial.h finds its sign
LeadTerm leadOfProductsDiff( const SosPoly & a, const SosPoly & b, const SosPoly & c, const SosPoly & d )
{
    // the terms of a*b (and of c*d) form rows: the i-th term of a times all the terms of b in the order of increasing degree;
    // the heap keeps the current term of every row, so all terms of both products are visited in the order of increasing degree
    struct RowTerm
    {
        std::int64_t deg;
        int i, j; // indices of the terms in the two factors
        bool neg; // true for the terms of c*d
    };
    auto greater = [] ( const RowTerm & x, const RowTerm & y ) { return x.deg > y.deg; };
    std::vector<RowTerm> heap;
    heap.reserve( a.get().size() + c.get().size() );
    if ( !b.get().empty() )
        for ( int i = 0; i < (int)a.get().size(); ++i )
            heap.push_back( { a.get()[i].first + b.get()[0].first, i, 0, false } );
    if ( !d.get().empty() )
        for ( int i = 0; i < (int)c.get().size(); ++i )
            heap.push_back( { c.get()[i].first + d.get()[0].first, i, 0, true } );
    std::make_heap( heap.begin(), heap.end(), greater );

    while ( !heap.empty() )
    {
        const auto deg = heap.front().deg;
        VarBigInt coeff;
        do
        {
            std::pop_heap( heap.begin(), heap.end(), greater );
            auto r = heap.back();
            heap.pop_back();
            const auto & f = ( r.neg ? c : a ).get();
            const auto & g = ( r.neg ? d : b ).get();
            const auto prod = f[r.i].second * g[r.j].second;
            if ( r.neg )
                coeff -= prod;
            else
                coeff += prod;
            if ( ++r.j < (int)g.size() )
            {
                r.deg = f[r.i].first + g[r.j].first;
                heap.push_back( r );
                std::push_heap( heap.begin(), heap.end(), greater );
            }
        }
        while ( !heap.empty() && heap.front().deg == deg );
        if ( const int s = coeff.sign() )
            return { deg, s };
    }
    return {};
}

/// whether perturbed point p comes before perturbed point q along the sweep, which orders the points by kx + eta * ky
/// with eps << eta; nullopt if they coincide after the perturbation too, or one of them is not defined
std::optional<bool> sosLess( const PreciseCutPoint & p, const PreciseCutPoint & q, int kx, int ky )
{
    SosRanks r;
    r.add( sosVerts( p ).data(), sosNumVerts( p ) );
    r.add( sosVerts( q ).data(), sosNumVerts( q ) );
    std::sort( r.ids.begin(), r.ids.begin() + r.n );
    for ( r.stage = 1; r.stage <= r.n; ++r.stage )
    {
        const auto a = sosPoint( p, r, kx, ky );
        const auto b = sosPoint( q, r, kx, ky );
        if ( !a || !b )
            return {};
        // the signs and the leading degrees of the differences of the coordinates of p and q
        const auto x = leadOfProductsDiff( a->nx, b->d, b->nx, a->d );
        const auto y = leadOfProductsDiff( a->ny, b->d, b->ny, a->d );
        if ( x.sign == 0 && y.sign == 0 )
            continue; // the vertices perturbed so far move both points alike
        // a difference that is still zero has a leading term of a later stage, so of a larger degree
        if ( x.sign != 0 && ( y.sign == 0 || x.deg <= y.deg ) )
            return x.sign < 0;
        return y.sign < 0;
    }
    return {};
}

/// the signs of the projected coordinates of the perturbed cross( normal of f, normal of g ), as the sweep along kx + eta * ky
/// with eps << eta sees them: a coordinate whose leading term has a larger degree than that of the other one is taken as zero
Vector2i sosCrossDirection( const std::array<PreciseVertCoords, 3> & f, const std::array<PreciseVertCoords, 3> & g, int kx, int ky )
{
    SosRanks r;
    r.add( f.data(), 3 );
    r.add( g.data(), 3 );
    std::sort( r.ids.begin(), r.ids.begin() + r.n );
    auto normal = [&] ( const std::array<PreciseVertCoords, 3> & t )
    {
        const auto a = sosCoords( t[0], r );
        return cross( sub( sosCoords( t[1], r ), a ), sub( sosCoords( t[2], r ), a ) );
    };
    auto lead = [] ( const SosPoly & p )
    {
        return p.empty() ? LeadTerm{} : LeadTerm{ p.get().front().first, p.get().front().second.sign() };
    };
    for ( r.stage = 1; r.stage <= r.n; ++r.stage )
    {
        const auto u = cross( normal( f ), normal( g ) );
        const auto x = lead( u[kx] ), y = lead( u[ky] );
        if ( x.sign == 0 && y.sign == 0 )
            continue;
        // a coordinate that is still zero has a leading term of a later stage, so of a larger degree
        return { y.sign == 0 || x.deg <= y.deg ? x.sign : 0, x.sign == 0 || y.deg <= x.deg ? y.sign : 0 };
    }
    return {};
}

} // anonymous namespace

bool CutTrianglePredicates::reset( const std::array<PreciseVertCoords, 3> & tri )
{
    tri_ = tri;
    approx_.clear();
    points_.clear();
    perturbedLess_.clear();
    failed_.store( false, std::memory_order_relaxed );

    normal_ = normalOf( tri );
    // drop the axis most aligned with the normal; order the two kept axes so that the projection keeps the orientation around the normal
    int dropAx = 0;
    for ( int i = 1; i < 3; ++i )
        if ( std::abs( normal_[i] ) > std::abs( normal_[dropAx] ) )
            dropAx = i;
    kx_ = ( dropAx + 1 ) % 3;
    ky_ = ( dropAx + 2 ) % 3;
    if ( normal_[dropAx] < 0 )
        std::swap( kx_, ky_ );
    return normal_[dropAx] != 0;
}

bool CutTrianglePredicates::addPoint( VertId v, const PreciseCutPoint & p )
{
    if ( points_.size() <= v )
    {
        points_.resize( v + 1 );
        approx_.resize( v + 1 );
    }
    auto & pt = points_[v];
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
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        return true;
    }
    const int i = corner( p.seg[0] ), j = corner( p.seg[1] );
    pt.sides = i >= 0 && j >= 0 ? 1 << ( j == ( i + 1 ) % 3 ? i : j ) : 0;
    pt.oa = volume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
    pt.ob = volume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
    assert( ( pt.oa <= 0 && pt.ob >= 0 ) || ( pt.oa >= 0 && pt.ob <= 0 ) ); // the segment reaches the plane
    if ( pt.oa == pt.ob )
    {
        // both are zero: the segment lies in the plane, where the crossing is defined only by the perturbation
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        fail_();
        return false;
    }
    // the point is a + s * ( b - a ), s = oa / ( oa - ob ), where oa and ob have different signs
    const double da = toDouble( pt.oa );
    const double s = da / ( da - toDouble( pt.ob ) );
    const Vector3d a( p.seg[0].pt ), b( p.seg[1].pt );
    approx_[v] = { a[kx_] + s * ( b[kx_] - a[kx_] ), a[ky_] + s * ( b[ky_] - a[ky_] ) };
    return true;
}

CutTrianglePredicates::Fraction CutTrianglePredicates::exactCoord_( VertId v, int axis ) const
{
    const auto & p = points_[v];
    if ( p.def.isVertex() )
        return { FastInt256( p.def.seg[0].pt[axis] ), FastInt256( 1 ) };
    // ( oa * b - ob * a ) / ( oa - ob ) for segment ab, with coordinates within 2^30 and volumes within 2^96
    Fraction res{
        Int128Mul256( p.oa ) * Int128Mul256( p.def.seg[1].pt[axis] ) - Int128Mul256( p.ob ) * Int128Mul256( p.def.seg[0].pt[axis] ),
        FastInt256( p.oa ) - FastInt256( p.ob ) };
    if ( res.d.sign() < 0 )
    {
        res.n = -res.n;
        res.d = -res.d;
    }
    return res;
}

int CutTrianglePredicates::compareExact_( VertId p, VertId q, int axis ) const
{
    const auto a = exactCoord_( p, axis );
    const auto b = exactCoord_( q, axis );
    const auto l = a.n * b.d;
    const auto r = b.n * a.d;
    return l < r ? -1 : ( r < l ? 1 : 0 );
}

int CutTrianglePredicates::sweepSign_( const Vector3i64 & u ) const
{
    if ( u[kx_] != 0 )
        return u[kx_] > 0 ? 1 : -1;
    return int( u[ky_] > 0 ) - int( u[ky_] < 0 );
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
    if ( const int c = compareExact_( p, q, kx_ ) )
        return c < 0;
    if ( a.y < b.y - cApproxTol )
        return true;
    if ( b.y < a.y - cApproxTol )
        return false;
    if ( const int c = compareExact_( p, q, ky_ ) )
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
        return sweepSign_( Vector3i64( s1.pt ) - Vector3i64( s0.pt ) ) > 0;
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
    const bool swapped = q < p;
    const auto key = swapped ? ( std::uint64_t( q ) << 32 ) | std::uint64_t( p ) : ( std::uint64_t( p ) << 32 ) | std::uint64_t( q );
    {
        std::lock_guard lock( perturbedLessMutex_ );
        if ( const auto it = perturbedLess_.find( key ); it != perturbedLess_.end() )
            return it->second != swapped;
    }
    const auto res = swapped ? lessByPerturbation( q, p ) : lessByPerturbation( p, q );
    if ( !res )
    {
        fail_(); // the points coincide after the perturbation too
        return p < q;
    }
    std::lock_guard lock( perturbedLessMutex_ );
    perturbedLess_[key] = *res;
    return *res != swapped;
}

std::optional<bool> CutTrianglePredicates::lessByPerturbation( VertId p, VertId q ) const
{
    return sosLess( points_[p].def, points_[q].def, kx_, ky_ );
}

std::optional<CutTrianglePredicates::Line> CutTrianglePredicates::commonLine_( VertId p, VertId q ) const
{
    const auto & a = points_[p];
    const auto & b = points_[q];
    if ( const int common = a.sides & b.sides )
        return Line{ .side = std::countr_zero( unsigned( common ) ) }; // one bit, unless p and q are the same vertex
    if ( a.def.isVertex() || b.def.isVertex() )
        return {};
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
    {
        const auto u = Vector3i64( tri_[( l.side + 1 ) % 3].pt ) - Vector3i64( tri_[l.side].pt );
        return { sgn( u[kx_] ), sgn( u[ky_] ) };
    }
    const auto n = normalOf( l.plane );
    const Vector2i res( crossComponentSign( normal_, n, kx_ ), crossComponentSign( normal_, n, ky_ ) );
    if ( res.x != 0 || res.y != 0 )
        return res;
    // only the perturbation defines the direction, e.g. of a cut by a triangle with a zero normal, whose crossings with
    // this triangle coincide before the perturbation
    return sosCrossDirection( tri_, l.plane, kx_, ky_ );
}

bool CutTrianglePredicates::along_( const Vector2i & dir, VertId p, VertId q ) const
{
    // q - p goes along the line, and the sweep meets p first iff it goes in the direction of the line,
    // whose projection is decided by its first coordinate, or by the second if the first is exactly zero, since eps << eta
    if ( dir.x == 0 && dir.y == 0 )
    {
        fail_(); // the line is not defined without the perturbation, so neither are the points on it
        return false;
    }
    return less( p, q ) == ( dir.x != 0 ? dir.x > 0 : dir.y > 0 );
}

bool CutTrianglePredicates::sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, const PreciseCutPoint & p ) const
{
    auto orient = [&] ( const PreciseVertCoords & x )
    {
        return orient3d( { plane[0], plane[1], plane[2], x } );
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
    const bool o0 = orient( p.seg[0] );
    const bool o1 = orient( p.seg[1] );
    if ( o0 == o1 )
        return o0;
    if ( sameTriangle( p.tri, plane ) )
    {
        fail_(); // the point is on the plane, so on the line
        return false;
    }
    // the segment crosses the plane: the point is on the side of the end it is closer to along the segment
    return segmentIntersectionTriPlaneOrder( { p.seg[0], p.seg[1], p.tri[0], p.tri[1], p.tri[2], plane[0], plane[1], plane[2] } ) ? o0 : o1;
}

bool CutTrianglePredicates::ccw( VertId a, VertId b, VertId c ) const
{
    if ( const int common = points_[a].sides & points_[b].sides & points_[c].sides )
    {
        // three points on one side: they bulge outward a little (infinitely less than the perturbation of the vertices),
        // so that the side is a strictly convex chain
        const auto dir = lineDirection_( Line{ .side = std::countr_zero( unsigned( common ) ) } );
        return int( along_( dir, a, b ) ) + int( along_( dir, b, c ) ) + int( along_( dir, c, a ) ) == 2;
    }
    // rotating the vertices keeps the orientation of the triangle, so any two of them on one line can come first
    const std::array<VertId, 3> v{ a, b, c };
    for ( int i = 0; i < 3; ++i )
    {
        const VertId p = v[i], q = v[( i + 1 ) % 3], r = v[( i + 2 ) % 3];
        const auto l = commonLine_( p, q );
        if ( !l )
            continue;
        const bool forward = along_( lineDirection_( *l ), p, q );
        if ( l->side >= 0 )
            return forward; // the triangle is on the left of its sides, and r is not on this one
        // the left of the line in its direction is the side of the plane opposite to the plane's normal, which is where orient3d is true
        return forward == sideOfPlane_( l->plane, points_[r].def );
    }
    fail_(); // no two of the points are on one line
    return false;
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
    if ( const int c = compareExact_( b, pivot, ky_ ) )
        return c < 0;
    if ( o.x < x.x - cApproxTol )
        return true;
    if ( x.x < o.x - cApproxTol )
        return false;
    if ( const int c = compareExact_( b, pivot, kx_ ) )
        return c > 0;
    // the points coincide before the perturbation: d goes along the line through them, defined without the perturbation
    const auto l = commonLine_( pivot, b );
    if ( !l )
    {
        fail_();
        return false;
    }
    auto dir = lineDirection_( *l );
    if ( !along_( dir, pivot, b ) )
        dir = -dir;
    return dir.y != 0 ? dir.y < 0 : dir.x > 0;
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} // namespace MR
