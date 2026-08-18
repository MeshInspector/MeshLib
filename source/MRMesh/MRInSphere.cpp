#include "MRInSphere.h"
#include "MRVarBigInt.h"
#include "MRInt64Mul128.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>


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

// variable-length big integer replacing the former boost::multiprecision::cpp_int; it grows with
// the value, so unlike FastInt<N> no per-sub-expression bit-width bounds are needed here
using BigInt = VarBigInt;

// value x + y * sqrt( ew ) for a fixed ew > 0; exact sign computations in inSphere tie resolution
struct SqrtNum
{
    BigInt x, y;
};

SqrtNum operator +( const SqrtNum & a, const SqrtNum & b ) { return { a.x + b.x, a.y + b.y }; }
SqrtNum operator -( const SqrtNum & a, const SqrtNum & b ) { return { a.x - b.x, a.y - b.y }; }

SqrtNum mul( const SqrtNum & a, const SqrtNum & b, const BigInt & ew )
{
    return { a.x * b.x + a.y * b.y * ew, a.x * b.y + a.y * b.x };
}

int signOf( const SqrtNum & a, const BigInt & ew )
{
    const int sx = a.x.sign();
    const int sy = a.y.sign();
    if ( sy == 0 || sx == sy )
        return sx;
    if ( sx == 0 )
        return sy;
    const BigInt l = a.x * a.x, r = a.y * a.y * ew;
    if ( l == r )
        return 0;
    return l > r ? sx : sy;
}

using SqrtVec = std::array<SqrtNum, 3>;

SqrtNum dot( const SqrtVec & u, const SqrtVec & v, const BigInt & ew )
{
    return mul( u[0], v[0], ew ) + mul( u[1], v[1], ew ) + mul( u[2], v[2], ew );
}

SqrtVec cross( const SqrtVec & u, const SqrtVec & v, const BigInt & ew )
{
    return SqrtVec{
        mul( u[1], v[2], ew ) - mul( u[2], v[1], ew ),
        mul( u[2], v[0], ew ) - mul( u[0], v[2], ew ),
        mul( u[0], v[1], ew ) - mul( u[1], v[0], ew ) };
}

// ---------- exact graded simulation-of-simplicity evaluation on symbolically perturbed points ----------
// used only when the not-perturbed sphere is degenerate: coincident or collinear triangle points
// (W == 0) or rSq exactly equal to the squared circumradius (E == 0); each coordinate becomes
// (value + eps^k), and the answer is given by the signs of the leading terms of the same predicate
// polynomials in eps

/// a term of the sparse polynomial in the perturbation parameter
struct EpsTerm
{
    std::int64_t deg = 0;
    BigInt coef;
};

/// sparse polynomial in the perturbation parameter: the terms are ordered by ascending degree
using EpsPoly = std::vector<EpsTerm>;

/// merges same-degree terms of a degree-sorted sequence, dropping the vanished ones
void mergeTerms( EpsPoly & r )
{
    size_t out = 0;
    for ( size_t i = 0; i < r.size(); )
    {
        auto deg = r[i].deg;
        auto coef = std::move( r[i].coef );
        for ( ++i; i < r.size() && r[i].deg == deg; ++i )
            coef += r[i].coef;
        if ( coef.sign() != 0 )
            r[out++] = { deg, std::move( coef ) };
    }
    r.resize( out );
}

EpsPoly operator +( const EpsPoly & a, const EpsPoly & b )
{
    EpsPoly r;
    r.reserve( a.size() + b.size() );
    std::merge( a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( r ),
        []( const EpsTerm & x, const EpsTerm & y ) { return x.deg < y.deg; } );
    mergeTerms( r );
    return r;
}

EpsPoly operator -( const EpsPoly & a, const EpsPoly & b )
{
    EpsPoly nb;
    nb.reserve( b.size() );
    for ( const auto & t : b )
        nb.push_back( { t.deg, -t.coef } );
    return a + nb;
}

EpsPoly operator *( const EpsPoly & a, const EpsPoly & b )
{
    EpsPoly r;
    r.reserve( a.size() * b.size() );
    for ( const auto & ta : a )
        for ( const auto & tb : b )
            r.push_back( { ta.deg + tb.deg, ta.coef * tb.coef } );
    std::sort( r.begin(), r.end(), []( const EpsTerm & x, const EpsTerm & y ) { return x.deg < y.deg; } );
    mergeTerms( r );
    return r;
}

/// the sign of the polynomial for eps -> +0, which is the sign of its lowest-degree term
int signOf( const EpsPoly & p )
{
    return p.empty() ? 0 : p.front().coef.sign();
}

using EpsVec = std::array<EpsPoly, 3>;

EpsPoly dot( const EpsVec & u, const EpsVec & v )
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

EpsVec cross( const EpsVec & u, const EpsVec & v )
{
    return {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0] };
}

/// the ladder between the ranks, larger than the maximal degree-weight of one point (8 * 9 = 72)
/// in the polynomials below, so the term order is lexicographic in the rank-weights and the
/// existence sign does not depend on whether the ranks are computed among 3 or among 4 points
constexpr std::int64_t cSosLadder = 128;

/// the base perturbation degrees: ladder^rank with the ranks by ascending ids
std::array<std::int64_t, 4> sosDegrees( const VertId * ids, int n )
{
    std::array<int, 4> order = { 0, 1, 2, 3 };
    std::sort( order.begin(), order.begin() + n, [&]( int l, int r ) { return ids[l] < ids[r]; } );
    std::array<std::int64_t, 4> res = {};
    std::int64_t deg = 1;
    for ( int i = 0; i < n; ++i, deg *= cSosLadder )
        res[order[i]] = deg;
    return res;
}

/// ( pTo + delta( degTo ) ) - ( pFrom + delta( degFrom ) ), where delta( d ) = ( eps^9d, eps^3d, eps^d )
EpsVec perturbedDiff( const Vector3i & pTo, const Vector3i & pFrom, std::int64_t degTo, std::int64_t degFrom )
{
    constexpr int axisMult[3] = { 9, 3, 1 };
    EpsVec r;
    for ( int i = 0; i < 3; ++i )
    {
        if ( const auto base = std::int64_t( pTo[i] ) - pFrom[i] )
            r[i].push_back( { 0, BigInt( base ) } );
        r[i].push_back( { axisMult[i] * degTo, BigInt( std::int64_t( 1 ) ) } );
        r[i].push_back( { axisMult[i] * degFrom, BigInt( std::int64_t( -1 ) ) } );
        std::sort( r[i].begin(), r[i].end(), []( const EpsTerm & x, const EpsTerm & y ) { return x.deg < y.deg; } );
    }
    return r;
}

struct SosQuantities
{
    EpsPoly W, E, A, t;
};

/// the predicate polynomials on the perturbed points; A and t are computed only for n == 4
SosQuantities buildSosQuantities( const Vector3i * pts, const VertId * ids, int n, std::int64_t rSq )
{
    const auto dg = sosDegrees( ids, n );
    const auto u = perturbedDiff( pts[1], pts[0], dg[1], dg[0] );
    const auto v = perturbedDiff( pts[2], pts[0], dg[2], dg[0] );
    const auto w = cross( u, v );
    SosQuantities res;
    res.W = dot( w, w );
    const auto uu = dot( u, u );
    const auto vv = dot( v, v );
    const auto uv = dot( u, v );
    const auto su = vv * ( uu - uv );
    const auto sv = uu * ( vv - uv );
    EpsVec M;
    for ( int i = 0; i < 3; ++i )
        M[i] = su * u[i] + sv * v[i];
    EpsPoly rSq4W2 = res.W * res.W;
    const BigInt rSq4 = BigInt( rSq ) * 4;
    for ( auto & t : rSq4W2 )
        t.coef = t.coef * rSq4;
    res.E = rSq4W2 - dot( M, M );
    if ( n == 4 )
    {
        const auto q = perturbedDiff( pts[3], pts[0], dg[3], dg[0] );
        res.A = res.W * dot( q, q ) - dot( q, M );
        res.t = dot( q, w );
    }
    return res;
}

/// resolves the existence of the sphere for the symbolically perturbed points of a degenerate triangle
bool sosSphereExists( const Vector3i * pts, const VertId * ids, std::int64_t rSq )
{
    const auto qs = buildSosQuantities( pts, ids, 3, rSq );
    assert( signOf( qs.W ) > 0 ); // the perturbed triangle is never degenerate for distinct ids
    const int sE = signOf( qs.E );
    assert( sE != 0 );
    return sE > 0;
}

/// the full evaluation of the predicate on the symbolically perturbed points
InSphereResult sosInSphereFull( const Vector3i * pts, const VertId * ids, std::int64_t rSq )
{
    const auto qs = buildSosQuantities( pts, ids, 4, rSq );
    assert( signOf( qs.W ) > 0 );
    const int sE = signOf( qs.E );
    assert( sE != 0 );
    if ( sE < 0 )
        return InSphereResult::NoSphere;
    const int sA = signOf( qs.A );
    const int st = signOf( qs.t );
    if ( sA < 0 && st >= 0 )
        return InSphereResult::Inside;
    if ( sA >= 0 && st <= 0 )
    {
        assert( sA != 0 || st != 0 ); // the perturbed query point is never exactly on the sphere
        return InSphereResult::Outside;
    }
    // the comparison A^2 W <> E t^2 by the leading terms: a leading term of a product is the
    // product of the leading terms, so the sign is known from the degrees alone unless they match
    // (the leading coefficients of W and E are positive here), and the full products are expanded
    // only on an exact tie of both the degrees and the coefficients
    const auto & [dA, cA] = qs.A.front();
    const auto & [dW, cW] = qs.W.front();
    const auto & [dE, cE] = qs.E.front();
    const auto & [dt, ct] = qs.t.front();
    int sG;
    if ( const auto degDiff = ( 2 * dA + dW ) - ( dE + 2 * dt ); degDiff != 0 )
        sG = degDiff < 0 ? 1 : -1;
    else
    {
        const auto coefX = cA * cA * cW;
        const auto coefY = cE * ( ct * ct );
        if ( coefX != coefY )
            sG = coefX > coefY ? 1 : -1;
        else
        {
            const auto G = qs.A * qs.A * qs.W - qs.E * ( qs.t * qs.t );
            sG = signOf( G );
            assert( sG != 0 );
        }
    }
    return ( ( sA < 0 ) == ( sG > 0 ) ) ? InSphereResult::Inside : InSphereResult::Outside;
}

} // anonymous namespace

bool InSphereTester<int>::reset( const Vector3i & va, const Vector3i & vb, const Vector3i & vc, std::int64_t sqRadius )
{
    // no overflow anywhere below given that any difference of two points is within +-0.99*2^31
    // (as getToIntConverter guarantees) and rSq fits in int64
    a = va;
    rSq = sqRadius;
    E = -1;
    u = Vector3i64{ vb - va };
    v = Vector3i64{ vc - va };

    // no sphere of radius sqrt(rSq) can pass via two points more than the diameter apart;
    // strictly greater: a side exactly equal to the diameter can lie on the sphere
    const auto rSq4 = 4 * FastInt128( sqRadius );
    const auto uu = dot( Vector3i64mul{ u }, Vector3i64mul{ u } );
    if ( uu > rSq4 )
        return false;
    const auto vv = dot( Vector3i64mul{ v }, Vector3i64mul{ v } );
    if ( vv > rSq4 )
        return false;
    const Vector3i64 bc = v - u;
    if ( dot( Vector3i64mul{ bc }, Vector3i64mul{ bc } ) > rSq4 )
        return false;

    w = cross( u, v ); // <= 2^63
    // the sum of three products of two 64-bit values does not fit in 128 bits
    W = FastInt<192>( Int64Mul128( w.x ) * Int64Mul128( w.x ) )
      + FastInt<192>( Int64Mul128( w.y ) * Int64Mul128( w.y ) )
      + FastInt<192>( Int64Mul128( w.z ) * Int64Mul128( w.z ) ); // <= 2^128
    if ( W == 0 )
        return false; // a, b, c are collinear => no circle through them

    // same as |u|^2 * cross( v, w ) + |v|^2 * cross( w, u ) expanded as in circumcircleCenter,
    // with one dot product instead of two cross products; components <= 2^161
    const auto uv = dot( Vector3i64mul{ u }, Vector3i64mul{ v } );
    const auto su = FastInt<192>( Int128Mul256( vv ) * Int128Mul256( uu - uv ) ); // <= 2^129
    const auto sv = FastInt<192>( Int128Mul256( uu ) * Int128Mul256( vv - uv ) );
    M = { FastInt<192>( su * u.x + sv * v.x ),
          FastInt<192>( su * u.y + sv * v.y ),
          FastInt<192>( su * u.z + sv * v.z ) };

    // negative: sqrt(rSq) is less than the circumradius of the triangle => no such sphere
    E = FastInt<384>( W * W * rSq * 4 - ( M[0] * M[0] + M[1] * M[1] + M[2] * M[2] ) ); // <= 2^322
    return E >= 0;
}

InSphereResult InSphereTester<int>::operator()( const Vector3i & d ) const
{
    assert( E >= 0 ); // the last reset() must have returned true
    const Vector3i64 q{ d - a };

    // d farther than the diameter from a point on the sphere is strictly outside
    const auto qq = dot( Vector3i64mul{ q }, Vector3i64mul{ q } );
    if ( qq > 4 * FastInt128( rSq ) )
        return InSphereResult::Outside;

    // A = |w|^2 * ( |d - circumcenter|^2 - sqr( circumradius ) ), <= 2^194
    const auto A = FastInt256( W * qq - ( M[0] * q.x + M[1] * q.y + M[2] * q.z ) );

    // t = |w| * signedDistance( d, plane of the triangle ), <= 2^96
    const auto t = dot( Vector3i64mul{ q }, Vector3i64mul{ w } );

    // d is strictly inside the sphere <=> A * |w| < sqrt( E ) * t
    if ( A < 0 && t >= 0 )
        return InSphereResult::Inside;
    if ( A >= 0 && t <= 0 )
        return ( A == 0 && ( t == 0 || E == 0 ) ) ? InSphereResult::OnSphere : InSphereResult::Outside;
    const auto lhs = FastInt<448>( A * A ) * W; // A * A <= 2^388, the product <= 2^514
    const auto rhs = E * sqr( Int128Mul256( t ) ); // <= 2^513
    if ( lhs == rhs )
        return InSphereResult::OnSphere;
    return ( A < 0 ) == ( lhs > rhs ) ? InSphereResult::Inside : InSphereResult::Outside;
}

InSphereResult inSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d, std::int64_t rSq )
{
    InSphereTesteri tester;
    if ( !tester.reset( a, b, c, rSq ) )
        return InSphereResult::NoSphere;
    return tester( d );
}

bool InSphereTesterSoS::reset( const PreciseVertCoords & va, const PreciseVertCoords & vb, const PreciseVertCoords & vc, std::int64_t sqRadius )
{
    va_ = va.id;
    vb_ = vb.id;
    vc_ = vc.id;
    degenerateTriangle_ = false;
    const Vector3i pts[3] = { va.pt, vb.pt, vc.pt };
    const VertId ids[3] = { va_, vb_, vc_ };
    if ( InSphereTester<int>::reset( va.pt, vb.pt, vc.pt, sqRadius ) )
    {
        if ( E > 0 )
            return true;
        // rSq is exactly equal to the squared circumradius: the perturbation decides the existence
        if ( sosSphereExists( pts, ids, sqRadius ) )
            return true;
        E = -1; // the perturbed sphere does not exist: poison the queries
        return false;
    }
    // sides beyond the diameter and rSq strictly below the squared circumradius are stable under
    // the perturbation, but a degenerate triangle (W == 0) must be resolved by it
    const auto rSq4 = 4 * FastInt128( sqRadius );
    if ( dot( Vector3i64mul{ u }, Vector3i64mul{ u } ) > rSq4
      || dot( Vector3i64mul{ v }, Vector3i64mul{ v } ) > rSq4 )
        return false;
    const Vector3i64 bc = v - u;
    if ( dot( Vector3i64mul{ bc }, Vector3i64mul{ bc } ) > rSq4 )
        return false;
    if ( cross( u, v ) != Vector3i64() )
        return false; // W > 0, so the base failed on E < 0, which is stable

    // closed-form resolutions of the degenerate triangles, validated against the full evaluation
    const bool sameAB = u == Vector3i64();
    const bool sameAC = v == Vector3i64();
    const bool sameBC = u == v;
    if ( int( sameAB ) + int( sameAC ) + int( sameBC ) >= 2 )
        return false; // all three coincide: the perturbed needle-like triangle has diverging circumradius
    if ( !( sameAB || sameAC || sameBC ) )
        return false; // distinct collinear points: the perturbed circumradius diverges as well
    // two coincident points separate along z after the perturbation, so the sphere exists iff
    // 4 rSq (Vx^2 + Vy^2) > |V|^4 at the leading order, V = the third point minus the pair
    // (V = -V for b == c, which does not change the rule)
    const Vector3i64 V = sameAB ? v : u;
    const auto vxy = FastInt128( Int64Mul128( V.x ) * Int64Mul128( V.x ) )
                   + FastInt128( Int64Mul128( V.y ) * Int64Mul128( V.y ) ); // <= 2^65
    const auto vv2 = dot( Vector3i64mul{ V }, Vector3i64mul{ V } );         // <= 2^66
    const auto lhs = FastInt<192>( Int128Mul256( vxy ) * Int128Mul256( 4 * FastInt128( sqRadius ) ) ); // <= 2^131
    const auto rhs = FastInt<192>( Int128Mul256( vv2 ) * Int128Mul256( vv2 ) ); // <= 2^132
    if ( lhs < rhs )
        return false;
    if ( lhs == rhs && !sosSphereExists( pts, ids, sqRadius ) )
        return false; // an exact tie of the leading rule is resolved by the full evaluation
    // the perturbed sphere converges to the sphere via the pair and the third point, tangent to
    // the z-axis at the pair; remember it for the closed-form queries, with the side of its center
    // given by the leading direction of the perturbed normal (validated against the full evaluation)
    if ( sameAB )
    {
        pairPt_ = a;
        pairV_ = v;
        pairSigma_ = vb_ < va_ ? 1 : -1;
    }
    else if ( sameAC )
    {
        pairPt_ = a;
        pairV_ = u;
        pairSigma_ = vc_ < va_ ? -1 : 1;
    }
    else
    {
        pairPt_ = a + Vector3i( u );
        pairV_ = -u;
        pairSigma_ = vc_ < vb_ ? 1 : -1;
    }
    degenerateTriangle_ = true;
    return true;
}

InSphereResult InSphereTesterSoS::operator()( const PreciseVertCoords & d ) const
{
    if ( degenerateTriangle_ || E == 0 )
    {
        if ( degenerateTriangle_ )
        {
            // strictly inside or outside the limit sphere resolves the query in closed form,
            // and only the points exactly on it need the full evaluation
            const Vector3i64 g{ d.pt - pairPt_ };
            const auto k = FastInt128( Int64Mul128( pairV_.x ) * Int64Mul128( pairV_.x ) )
                         + FastInt128( Int64Mul128( pairV_.y ) * Int64Mul128( pairV_.y ) );
            const auto ss = k + FastInt128( Int64Mul128( pairV_.z ) * Int64Mul128( pairV_.z ) );
            const BigInt bk{ k }, bs{ ss };
            const BigInt D = 4 * BigInt{ rSq } * bk - bs * bs; // >= 0 since the sphere exists
            const auto gg = dot( Vector3i64mul{ g }, Vector3i64mul{ g } );
            const auto gvxy = FastInt128( Int64Mul128( g.x ) * Int64Mul128( pairV_.x ) )
                            + FastInt128( Int64Mul128( g.y ) * Int64Mul128( pairV_.y ) );
            const auto gcrs = FastInt128( Int64Mul128( g.y ) * Int64Mul128( pairV_.x ) )
                            - FastInt128( Int64Mul128( g.x ) * Int64Mul128( pairV_.y ) );
            // |g - center|^2 - rSq multiplied by k: ( k |g|^2 - s (g.V)xy ) - sigma sqrt(D) ( gy Vx - gx Vy )
            const SqrtNum val{ bk * BigInt{ gg } - bs * BigInt{ gvxy },
                BigInt{ std::int64_t( -pairSigma_ ) } * BigInt{ gcrs } };
            const int sg = D.sign() > 0 ? signOf( val, D ) : val.x.sign(); // D == 0 on the existence tie
            if ( sg )
                return sg < 0 ? InSphereResult::Inside : InSphereResult::Outside;
        }
        else
        {
            // an exact rSq == squared circumradius: only the "exactly on the sphere" queries
            // reach the full evaluation
            const auto res = InSphereTester<int>::operator()( d.pt );
            if ( res != InSphereResult::OnSphere )
                return res;
        }
        // b and c are reconstructed exactly from the stored differences
        const Vector3i pts[4] = { a, a + Vector3i( u ), a + Vector3i( v ), d.pt };
        const VertId ids[4] = { va_, vb_, vc_, d.id };
        const auto res = sosInSphereFull( pts, ids, rSq );
        assert( res != InSphereResult::NoSphere ); // the existence was resolved positively in reset()
        return res;
    }
    const auto res = InSphereTester<int>::operator()( d.pt );
    if ( res != InSphereResult::OnSphere )
        return res;

    // the sphere points with their ids; b and c are reconstructed exactly from the stored differences
    const PreciseVertCoords vs[4] = { { va_, a }, { vb_, a + Vector3i( u ) }, { vc_, a + Vector3i( v ) }, d };
#ifndef NDEBUG
    for ( int i = 0; i < 3; ++i )
        for ( int j = i + 1; j < 4; ++j )
            assert( vs[i].id != vs[j].id );
#endif

    // vs[3] is exactly on the sphere: perturb the points in the order of ascending ids, the first point
    // whose perturbation moves vs[3] off the sphere decides; when a triangle point is perturbed, the
    // center moves along the circle of points equidistant from two other triangle points, on which
    // the squared distance to vs[3] is a degree-1 trigonometric polynomial
    const BigInt bigW = BigInt( W );
    const BigInt ew = BigInt( E ) * bigW; // sqr( S ); the signs below are computed in Z[S]

    // ch[k] = 2 W^2 * ( vs[k].pt - sphereCenter ) = 2 W^2 ( vs[k].pt - vs[0].pt ) - W M - S w
    const Vector3i64 rel[4] = { {}, u, v, Vector3i64{ d.pt - a } };
    std::array<SqrtVec, 4> ch;
    for ( int k = 0; k < 4; ++k )
        for ( int i = 0; i < 3; ++i )
            ch[k][i] = { 2 * bigW * bigW * rel[k][i] - bigW * BigInt( M[i] ), -BigInt{ w[i] } };

    int order[4] = { 0, 1, 2, 3 };
    std::sort( std::begin( order ), std::end( order ), [&]( int l, int r ) { return vs[l].id < vs[r].id; } );

    for ( int idx : order )
    {
        if ( idx == 3 )
        {
            // perturbing the query point never moves the sphere: its largest (z) perturbation resolves
            // the tie, and an exactly tangential move exits the sphere
            return signOf( ch[3][2], ew ) < 0 ? InSphereResult::Inside : InSphereResult::Outside;
        }
        const int q1 = idx == 0 ? 1 : 0;
        const int q2 = idx == 2 ? 1 : 2;
        if ( vs[q1].pt == vs[3].pt || vs[q2].pt == vs[3].pt )
            continue; // idle point: its perturbation keeps vs[3] exactly on the sphere
        const auto T = cross( ch[q1], ch[q2], ew );
        const int sG = signOf( dot( ch[3], T, ew ), ew );
        const int sN = signOf( dot( ch[idx], T, ew ), ew );
        assert( sN != 0 ); // E > 0 means the center is not in the plane of the triangle
        if ( sG != 0 )
        {
            const int sz = signOf( ch[idx][2], ew );
            return ( sz != 0 ? sG * sz * sN : sG * sN ) > 0 ? InSphereResult::Inside : InSphereResult::Outside;
        }
        // the first derivative vanishes for all perturbation directions of vs[idx]; the answer is given
        // by the curvature term (D-c)*(c-m) + sqr(rho), m = middle of two other triangle points,
        // rho = radius of the circle of centers
        SqrtVec chm;
        for ( int i = 0; i < 3; ++i )
            chm[i] = { bigW * BigInt( M[i] ) - bigW * bigW * ( BigInt{ rel[q1][i] } + rel[q2][i] ), BigInt{ w[i] } };
        const Vector3i64 qr{ vs[q1].pt - vs[q2].pt };
        auto f2 = dot( ch[3], chm, ew );
        // + W^4 * ( 4 rSq - |qr|^2 ): qr <= 2^32, so |qr|^2 <= 2^66 fits in 128 bits as in the primary path
        const auto qrSq = dot( Vector3i64mul{ qr }, Vector3i64mul{ qr } ); // FastInt128
        f2.x += bigW * bigW * bigW * bigW * ( 4 * BigInt{ rSq } - BigInt{ qrSq } );
        if ( const int sF = signOf( f2, ew ) )
            return sF < 0 ? InSphereResult::Inside : InSphereResult::Outside;
        assert( false ); // possible only for an idle point, and those are skipped above
    }
    assert( false ); // the query point always resolves the tie
    return InSphereResult::Outside;
}

bool FastInSphereTesterSoS::reset( const PreciseVertCoords & va, const PreciseVertCoords & vb, const PreciseVertCoords & vc, std::int64_t sqRadius )
{
    if ( !InSphereTesterSoS::reset( va, vb, vc, sqRadius ) )
        return false;

    if ( degenerateTriangle_ )
    {
        // only the perturbed sphere exists, there is no center to compute:
        // every query goes via the full symbolic evaluation of the base class
        cc_ = hn_ = {};
        tol_ = 0;
        return true;
    }

    // circumcenter - a = M / (2W) lies in the plane of the triangle and the height S * w / (2W^2)
    // is orthogonal to it, so the sum below cannot lose precision to cancellation;
    // W, M, E cannot be recomputed in double: E = 4 rSq W^2 - |M|^2 is a difference of two values
    // below 2^322 that vanishes exactly when the circumradius reaches the radius, so it cancels
    // completely on the near-degenerate triangles this filter must get right - converting the
    // exact values is both the cheapest and the only accurate source
    const auto dW = toDouble( W );
    cc_ = Vector3d{ toDouble( M[0] ), toDouble( M[1] ), toDouble( M[2] ) } / ( 2 * dW );
    hn_ = ( std::sqrt( toDouble( E ) * dW ) / ( 2 * dW * dW ) ) * Vector3d( w );

    // the tolerance of the tests in operator() and outsideBothSpheres, with u = 2^-53 and r the
    // radius: |cc_| is the circumradius and |hn_| the height, both at most r, and each carries a few
    // relative u - one from toDouble, which is correctly rounded, the rest from the operations above,
    // including the conversion of w, whose components reach 2^63 and are NOT exact in double. That
    // puts the center about 10 * u * r away from the true one. Near the surface, the only place where
    // the tolerance decides anything, |q - center| is about r, so the squared distance is off by
    //   2 * r * 10 * u * r  from the center, plus the roundings of three squares, two sums, the
    //   conversion of rSq and the subtraction, together about 34 * u * rSq < rSq * 2^-48,
    // and the tolerance takes 16 times that. Measured worst over 60k queries placed within 3e-14 of
    // the surface: rSq * 2^-50, no query decided against the exact predicate.
    // The tolerance is proportional to rSq rather than absolute, which is what makes it sound in both
    // directions: where the exact value is not positive, |q - center|^2 <= rSq bounds the error below
    // the tolerance, so the point is never called Outside; where it is not negative, |q - center|^2 >=
    // rSq bounds the computed value above -tolerance, so it is never called Inside. An absolute
    // tolerance fitted to the largest coordinates would instead swallow the whole range at small ones.
    tol_ = double( rSq ) * 0x1p-44;
    return true;
}

InSphereResult FastInSphereTesterSoS::operator()( const PreciseVertCoords & d ) const
{
    if ( degenerateTriangle_ )
        return InSphereTesterSoS::operator()( d ); // no exact sphere to filter against
    assert( E >= 0 ); // the last reset() must have returned true
    // sqr( distance from d to the center ) - rSq, which is negative exactly when d is inside
    const auto e = ( Vector3d( Vector3i64{ d.pt - a } ) - cc_ - hn_ ).lengthSq() - double( rSq );
    if ( e > tol_ )
        return InSphereResult::Outside;
    if ( e < -tol_ )
        return InSphereResult::Inside;
    return InSphereTesterSoS::operator()( d );
}

bool FastInSphereTesterSoS::outsideBothSpheres( const Vector3i & d ) const
{
    if ( degenerateTriangle_ )
        return false; // the answer for the perturbed spheres would need the id of d: never prune
    assert( E >= 0 ); // the last reset() must have returned true
    const Vector3i64 q{ d - a };

    // the two centers are cc_ + hn_ and cc_ - hn_, mirror images in the plane of the triangle
    const auto qd = Vector3d( q ) - cc_;
    const auto e0 = ( qd - hn_ ).lengthSq() - double( rSq );
    const auto e1 = ( qd + hn_ ).lengthSq() - double( rSq );
    if ( e0 > tol_ && e1 > tol_ )
        return true;
    if ( e0 < -tol_ || e1 < -tol_ )
        return false;

    // d farther than the diameter from a point on the spheres is strictly outside both of them
    const auto qq = dot( Vector3i64mul{ q }, Vector3i64mul{ q } );
    if ( qq > 4 * FastInt128( rSq ) )
        return true;

    // outside both means A * |w| > sqrt( E ) * | t | in the notation of operator(),
    // with the same bounds on A and t there
    const auto A = FastInt256( W * qq - ( M[0] * q.x + M[1] * q.y + M[2] * q.z ) );
    if ( A <= 0 )
        return false; // d is inside the selected sphere, or on it
    const auto t = dot( Vector3i64mul{ q }, Vector3i64mul{ w } );
    return FastInt<448>( A * A ) * W > E * sqr( Int128Mul256( t ) );
}

InSphereResult inSphere( const std::array<PreciseVertCoords, 4> & vs, std::int64_t rSq )
{
    InSphereTesterSoS tester;
    if ( !tester.reset( vs[0], vs[1], vs[2], rSq ) )
        return InSphereResult::NoSphere;
    return tester( vs[3] );
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} //namespace MR
