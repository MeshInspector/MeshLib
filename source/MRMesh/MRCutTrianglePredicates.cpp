#include "MRCutTrianglePredicates.h"
#include "MRInt64Mul128.h"
#include "MRSparsePolynomial.h"
#include "MRVarBigInt.h"
#include <algorithm>
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

// the rounded projected coordinates of a point are within 2^-19 integer units from the exact ones: for an edge-triangle
// point, a ratio of two volumes converted in double, times a coordinate difference within 2^31; for a crossing,
// two exactly computed values converted and divided; so a larger difference of two of them decides
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

/// the plane dot( n, x ) == d of a triangle
struct ExactPlane
{
    Vector3i64 n; // the components are within 2^63, since every coordinate difference is within 2^31
    FastInt128 d; // within 2^95
};

ExactPlane exactPlane( const std::array<PreciseVertCoords, 3> & t )
{
    const Vector3i64 a( t[0].pt );
    ExactPlane res;
    res.n = cross( Vector3i64( t[1].pt ) - a, Vector3i64( t[2].pt ) - a );
    res.d = Int64Mul128( res.n.x ) * Int64Mul128( a.x ) + Int64Mul128( res.n.y ) * Int64Mul128( a.y ) + Int64Mul128( res.n.z ) * Int64Mul128( a.z );
    return res;
}

/// component k of cross( a, b ) for vectors with components within 2^63, which is within 2^127
FastInt<192> crossComponent( const Vector3i64 & a, const Vector3i64 & b, int k )
{
    const int i = ( k + 1 ) % 3, j = ( k + 2 ) % 3;
    return FastInt<192>( Int64Mul128( a[i] ) * Int64Mul128( b[j] ) ) - FastInt<192>( Int64Mul128( a[j] ) * Int64Mul128( b[i] ) );
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
// gets eps^( w[i] * 128^r ), w = 9, 3, 1 for x, y, z, as in getPointDegrees of MRPrecisePredicates3.cpp with 128
// in place of 27. No vertex weighs 128 or more in a term below (13 coordinates at most, each of weight 9 at most),
// so the terms are ordered by the weights of the vertices from the largest id down, which is also how orient3d orders
// them: this is the perturbation of the boolean. A degree stores the weight of rank r in bits 7r..7r+6.
// The vertices are perturbed in stages: at stage s only the s smallest ids move, so the polynomials stay small, and
// a polynomial not zero at stage s already has the leading term of the full one, since all terms involving the other
// vertices come after it.

constexpr int cSosBits = 7;
constexpr int cMaxSosStage = 9; // 9 * 7 = 63 bits of a degree

using SosPoly = SparsePolynomial<VarBigInt, std::int64_t, std::numeric_limits<std::int64_t>::max()>;
using SosVec = std::array<SosPoly, 3>;

/// the vertices of one comparison, by ascending ids, of which the first `stage` ones are perturbed
struct SosRanks
{
    std::array<VertId, 16> ids;
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

/// a point as the perturbation needs it: the vertices defining it
struct SosDef
{
    enum class Kind
    {
        Vertex, // vs[0]
        EdgeTri, // the segment vs[0], vs[1] crossing the plane of triangle vs[2], vs[3], vs[4]
        Planes // the common point of the planes of triangles vs[0..2], vs[3..5], vs[6..8]
    } kind = Kind::Vertex;
    std::array<PreciseVertCoords, 9> vs;

    int numVerts() const { return kind == Kind::Vertex ? 1 : ( kind == Kind::EdgeTri ? 5 : 9 ); }
};

/// a point with perturbed coordinates: ( nx / d, ny / d ) in the projection
struct SosFraction
{
    SosPoly nx, ny, d;
};

/// the perturbed projected point, or nullopt if it is not defined before the perturbation
std::optional<SosFraction> sosPoint( const SosDef & def, const SosRanks & r, int kx, int ky )
{
    SosFraction res;
    if ( def.kind == SosDef::Kind::Vertex )
    {
        auto c = sosCoords( def.vs[0], r );
        res = { std::move( c[kx] ), std::move( c[ky] ), sosConst( 1 ) };
    }
    else if ( def.kind == SosDef::Kind::EdgeTri )
    {
        const auto a = sosCoords( def.vs[0], r ), b = sosCoords( def.vs[1], r );
        const auto t0 = sosCoords( def.vs[2], r ), t1 = sosCoords( def.vs[3], r ), t2 = sosCoords( def.vs[4], r );
        auto volume = [&] ( const SosVec & x ) { return dot( sub( t0, x ), cross( sub( t1, x ), sub( t2, x ) ) ); };
        const auto oa = volume( a ), ob = volume( b );
        // ( oa * b - ob * a ) / ( oa - ob ), as in CutTrianglePredicates::exactCoord_
        res = { oa * b[kx] - ob * a[kx], oa * b[ky] - ob * a[ky], oa - ob };
    }
    else
    {
        struct Plane
        {
            SosVec n;
            SosPoly d;
        };
        auto plane = [&] ( int i )
        {
            const auto p0 = sosCoords( def.vs[i], r ), p1 = sosCoords( def.vs[i + 1], r ), p2 = sosCoords( def.vs[i + 2], r );
            Plane res;
            res.n = cross( sub( p1, p0 ), sub( p2, p0 ) );
            res.d = dot( res.n, p0 );
            return res;
        };
        const auto f = plane( 0 ), g = plane( 3 ), h = plane( 6 );
        // ( df * cross( ng, nh ) + dg * cross( nh, nf ) + dh * cross( nf, ng ) ) / dot( nf, cross( ng, nh ) ), as in CutTrianglePredicates::addCrossing
        const auto cgh = cross( g.n, h.n ), chf = cross( h.n, f.n ), cfg = cross( f.n, g.n );
        res = { f.d * cgh[kx] + g.d * chf[kx] + h.d * cfg[kx], f.d * cgh[ky] + g.d * chf[ky] + h.d * cfg[ky], dot( f.n, cgh ) };
    }
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
std::optional<bool> sosLess( const SosDef & p, const SosDef & q, int kx, int ky )
{
    SosRanks r;
    r.add( p.vs.data(), p.numVerts() );
    r.add( q.vs.data(), q.numVerts() );
    std::sort( r.ids.begin(), r.ids.begin() + r.n );
    const int maxStage = std::min( r.n, cMaxSosStage );
    for ( r.stage = 1; r.stage <= maxStage; ++r.stage )
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

} // anonymous namespace

bool CutTrianglePredicates::reset( const std::array<PreciseVertCoords, 3> & tri )
{
    tri_ = tri;
    approx_.clear();
    points_.clear();
    crossings_.clear();
    perturbedLess_.clear();
    failed_.store( false, std::memory_order_relaxed );

    const Vector3i64 a( tri[0].pt );
    normal_ = cross( Vector3i64( tri[1].pt ) - a, Vector3i64( tri[2].pt ) - a );
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
    pt.crossing = -1;
    if ( p.isVertex() )
    {
        approx_[v] = { double( p.seg[0].pt[kx_] ), double( p.seg[0].pt[ky_] ) };
        return true;
    }
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

bool CutTrianglePredicates::addCrossing( VertId v, const PreciseCutLine & a, const PreciseCutLine & b, const ConvertToFloatVector & toFloat )
{
    assert( !a.isSide() && !b.isSide() );
    if ( points_.size() <= v )
    {
        points_.resize( v + 1 );
        approx_.resize( v + 1 );
    }
    points_[v].crossing = int( crossings_.size() );
    auto & c = crossings_.emplace_back();
    c.a = a.plane;
    c.b = b.plane;

    // by Cramer's rule, the common point of three planes dot( n, x ) == d is
    // ( df * cross( ng, nh ) + dg * cross( nh, nf ) + dh * cross( nf, ng ) ) / dot( nf, cross( ng, nh ) )
    const auto f = exactPlane( tri_ ), g = exactPlane( a.plane ), h = exactPlane( b.plane );
    std::array<FastInt<192>, 3> cgh;
    for ( int k = 0; k < 3; ++k )
        cgh[k] = crossComponent( g.n, h.n, k );
    FastInt256 d = FastInt256( cgh[0] * f.n.x ) + FastInt256( cgh[1] * f.n.y ) + FastInt256( cgh[2] * f.n.z ); // within 2^192
    std::array<FastInt256, 3> n;
    for ( int k = 0; k < 3; ++k )
        n[k] = FastInt256( f.d * cgh[k] + g.d * crossComponent( h.n, f.n, k ) + h.d * crossComponent( f.n, g.n, k ) ); // within 2^224
    if ( d.sign() == 0 )
    {
        approx_[v] = {};
        fail_();
        return false;
    }
    if ( d.sign() < 0 )
    {
        d = -d;
        for ( auto & x : n )
            x = -x;
    }
    // the same exact values in each of the three triangles, so the same rounding
    const double dd = toDouble( d );
    const Vector3d pos( toDouble( n[0] ) / dd, toDouble( n[1] ) / dd, toDouble( n[2] ) / dd );
    c.nx = n[kx_];
    c.ny = n[ky_];
    c.d = d;
    auto toInt = [] ( double x )
    {
        return int( std::round( std::clamp( x, -0x1p30, 0x1p30 ) ) );
    };
    c.pos = toFloat( Vector3i( toInt( pos.x ), toInt( pos.y ), toInt( pos.z ) ) );
    // beyond the integer range, where no crossing inside the triangle is, the rounding error can exceed the tolerance of less:
    // a not-a-number there makes it compare exactly
    const bool outside = std::abs( pos[kx_] ) > 0x1p31 || std::abs( pos[ky_] ) > 0x1p31;
    approx_[v] = outside ? Vector2d( std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() ) : Vector2d( pos[kx_], pos[ky_] );
    return true;
}

Vector3f CutTrianglePredicates::crossingPos( VertId v ) const
{
    assert( points_[v].crossing >= 0 );
    return crossings_[points_[v].crossing].pos;
}

CutTrianglePredicates::Fraction CutTrianglePredicates::exactCoord_( VertId v, int axis ) const
{
    const auto & p = points_[v];
    if ( p.crossing >= 0 )
    {
        const auto & c = crossings_[p.crossing];
        return { axis == kx_ ? c.nx : c.ny, c.d };
    }
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
    const auto & a = points_[p];
    const auto & b = points_[q];
    if ( a.crossing < 0 && b.crossing < 0 )
    {
        const auto & da = a.def;
        const auto & db = b.def;
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
        fail_(); // the points coincide after the perturbation too, or so many vertices move them alike that the stages ran out
        return p < q;
    }
    std::lock_guard lock( perturbedLessMutex_ );
    perturbedLess_[key] = *res;
    return *res != swapped;
}

std::optional<bool> CutTrianglePredicates::lessByPerturbation( VertId p, VertId q ) const
{
    auto def = [&] ( VertId v )
    {
        SosDef res;
        const auto & pt = points_[v];
        if ( pt.crossing >= 0 )
        {
            const auto & c = crossings_[pt.crossing];
            res.kind = SosDef::Kind::Planes;
            std::copy( tri_.begin(), tri_.end(), res.vs.begin() );
            std::copy( c.a.begin(), c.a.end(), res.vs.begin() + 3 );
            std::copy( c.b.begin(), c.b.end(), res.vs.begin() + 6 );
        }
        else if ( pt.def.isVertex() )
            res.vs[0] = pt.def.seg[0];
        else
        {
            res.kind = SosDef::Kind::EdgeTri;
            res.vs[0] = pt.def.seg[0];
            res.vs[1] = pt.def.seg[1];
            std::copy( pt.def.tri.begin(), pt.def.tri.end(), res.vs.begin() + 2 );
        }
        return res;
    };
    return sosLess( def( p ), def( q ), kx_, ky_ );
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

bool CutTrianglePredicates::leftOf( const PreciseCutLine & l, bool reversed, VertId p ) const
{
    // a side bounds the triangle, which is on its left; the points on the same side are taken as bulging outward a little
    // (infinitely less than the perturbation of the vertices), so that the sides are strictly convex chains
    if ( l.isSide() )
        return !reversed;
    assert( points_[p].crossing < 0 );
    // inside the triangle, the line is its crossing with the plane, and the left of cross( normal, plane normal ) is the side
    // of the plane opposite to its normal, which is where orient3d is true
    return sideOfPlane_( l.plane, points_[p].def ) == ( l.alongCross != reversed );
}

bool CutTrianglePredicates::ccwFromBehind( const PreciseCutLine & l, bool reversed ) const
{
    // the direction of the line, of which only the signs of the projected coordinates matter
    int sx, sy;
    if ( l.isSide() )
    {
        const auto u = Vector3i64( tri_[( l.side + 1 ) % 3].pt ) - Vector3i64( tri_[l.side].pt );
        sx = int( u[kx_] > 0 ) - int( u[kx_] < 0 );
        sy = int( u[ky_] > 0 ) - int( u[ky_] < 0 );
    }
    else
    {
        const auto n = exactPlane( l.plane ).n;
        sx = crossComponentSign( normal_, n, kx_ );
        sy = crossComponentSign( normal_, n, ky_ );
        if ( !l.alongCross )
        {
            sx = -sx;
            sy = -sy;
        }
    }
    if ( reversed )
    {
        sx = -sx;
        sy = -sy;
    }
    // the triangle ( o - R * sweep, x, o ) for the sweep direction ( 1, eta ), with x - o along the direction u:
    // its doubled area is R * ( eta * u.x - u.y ), and only an exactly zero u.y leaves u.x to decide, since eps << eta
    assert( sx != 0 || sy != 0 );
    return sy != 0 ? sy < 0 : sx > 0;
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} // namespace MR
