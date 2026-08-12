#include "MRInSphere.h"
#include "MRHighPrecision.h"
#include "MRInt64Mul128.h"
#include <algorithm>
#include <cassert>

namespace MR
{

#if __GNUC__ >= 12 // false positive array-bounds warnings in boost widening conversions like Int1024( Int256 )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

namespace
{

using BigInt = boost::multiprecision::cpp_int;

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
    return InSphereTester<int>::reset( va.pt, vb.pt, vc.pt, sqRadius );
}

InSphereResult InSphereTesterSoS::operator()( const PreciseVertCoords & d ) const
{
    const auto res = InSphereTester<int>::operator()( d.pt );
    if ( res != InSphereResult::OnSphere )
        return res;
    if ( E == 0 )
        return InSphereResult::Outside; // a perturbation of the triangle breaks the sphere's existence here

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
    const BigInt bigW = toBoostInt<BigInt>( W );
    const BigInt ew = toBoostInt<BigInt>( E ) * bigW; // sqr( S ); the signs below are computed in Z[S]

    // ch[k] = 2 W^2 * ( vs[k].pt - sphereCenter ) = 2 W^2 ( vs[k].pt - vs[0].pt ) - W M - S w
    const Vector3i64 rel[4] = { {}, u, v, Vector3i64{ d.pt - a } };
    std::array<SqrtVec, 4> ch;
    for ( int k = 0; k < 4; ++k )
        for ( int i = 0; i < 3; ++i )
            ch[k][i] = { 2 * bigW * bigW * rel[k][i] - bigW * toBoostInt<BigInt>( M[i] ), -BigInt{ w[i] } };

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
            chm[i] = { bigW * toBoostInt<BigInt>( M[i] ) - bigW * bigW * ( BigInt{ rel[q1][i] } + rel[q2][i] ), BigInt{ w[i] } };
        const Vector3i64 qr{ vs[q1].pt - vs[q2].pt };
        auto f2 = dot( ch[3], chm, ew );
        f2.x += bigW * bigW * bigW * bigW * ( 4 * BigInt{ rSq } - BigInt{ dot( Vector3i128{ qr }, Vector3i128{ qr } ) } );
        if ( const int sF = signOf( f2, ew ) )
            return sF < 0 ? InSphereResult::Inside : InSphereResult::Outside;
        assert( false ); // possible only for an idle point, and those are skipped above
    }
    assert( false ); // the query point always resolves the tie
    return InSphereResult::Outside;
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
