#include "MRInSphere.h"
#include "MRHighPrecision.h"
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

// what inSphere computes, exposed for the tie resolution in the simulation-of-simplicity flavor
struct InSphereQuantities
{
    Vector3i64 u, v, q; // B - A, C - A, D - A
    Vector3i64 w;       // doubled normal of triangle ABC
    Int256 W;           // |w|^2
    Vector3i256 M;      // 2 * |w|^2 * ( circumcenter(ABC) - A )
    Int512 E;           // sqr( 2 * h * |w|^2 ), h = distance from plane ABC to the sphere's center
};

/// returns +1 if D is strictly inside the sphere, 0 if D is exactly on it (and E > 0), -1 otherwise
int classifyInSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d,
    std::int64_t rSq, InSphereQuantities & qs )
{
    // no overflow anywhere below given that any difference of two points is within +-0.99*2^31
    // (as getToIntConverter guarantees) and rSq fits in int64

    qs.u = Vector3i64{ b - a };
    qs.v = Vector3i64{ c - a };
    qs.q = Vector3i64{ d - a };

    qs.w = cross( qs.u, qs.v ); // <= 2^63
    qs.W = dot( Vector3i256{ qs.w }, Vector3i256{ qs.w } ); // <= 2^128
    if ( qs.W == 0 )
        return -1; // A, B, C are collinear => no circle through them

    // components <= 2^160
    const auto uu = dot( Vector3i128{ qs.u }, Vector3i128{ qs.u } );
    const auto vv = dot( Vector3i128{ qs.v }, Vector3i128{ qs.v } );
    qs.M = Int256( uu ) * Vector3i256{ cross( Vector3i128{ qs.v }, Vector3i128{ qs.w } ) }
         + Int256( vv ) * Vector3i256{ cross( Vector3i128{ qs.w }, Vector3i128{ qs.u } ) };

    qs.E = 4 * Int512( rSq ) * sqr( Int512( qs.W ) ) - dot( Vector3i512{ qs.M }, Vector3i512{ qs.M } ); // <= 2^321
    if ( qs.E < 0 )
        return -1; // sqrt(rSq) is less than the circumradius of ABC => no such sphere

    // on-sphere ties with rSq exactly equal to the squared circumradius are reported as -1 (outside),
    // since a perturbation of A, B, C breaks the sphere's existence there
    const int onSphere = qs.E == 0 ? -1 : 0;

    // A = |w|^2 * ( |D - circumcenter(ABC)|^2 - sqr( circumradius ) ), <= 2^193
    const auto A = qs.W * Int256( dot( Vector3i128{ qs.q }, Vector3i128{ qs.q } ) ) - dot( Vector3i256{ qs.q }, qs.M );

    // t = |w| * signedDistance( D, plane ABC ), <= 2^96
    const auto t = dot( Vector3i128{ qs.q }, Vector3i128{ qs.w } );

    // D is strictly inside the sphere <=> A * |w| < sqrt( E ) * t
    if ( A < 0 && t >= 0 )
        return 1;
    if ( A >= 0 && t <= 0 )
        return ( A == 0 && ( t == 0 || qs.E == 0 ) ) ? onSphere : -1;
    const auto lhs = sqr( Int1024( A ) ) * Int1024( qs.W ); // <= 2^513
    const auto rhs = Int1024( qs.E ) * sqr( Int1024( t ) ); // <= 2^512
    if ( lhs == rhs )
        return onSphere;
    return ( A < 0 ) == ( lhs > rhs ) ? 1 : -1;
}

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

bool inSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d, std::int64_t rSq )
{
    InSphereQuantities qs;
    return classifyInSphere( a, b, c, d, rSq, qs ) > 0;
}

bool inSphere( const std::array<PreciseVertCoords, 4> & vs, std::int64_t rSq )
{
#ifndef NDEBUG
    for ( int i = 0; i < 3; ++i )
        for ( int j = i + 1; j < 4; ++j )
            assert( vs[i].id != vs[j].id );
#endif
    InSphereQuantities qs;
    if ( const int res = classifyInSphere( vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt, rSq, qs ) )
        return res > 0;

    // vs[3] is exactly on the sphere: perturb the points in the order of ascending ids, the first point
    // whose perturbation moves vs[3] off the sphere decides; when a triangle point is perturbed, the
    // center moves along the circle of points equidistant from two other triangle points, on which
    // the squared distance to vs[3] is a degree-1 trigonometric polynomial
    const BigInt W{ qs.W };
    const BigInt ew = BigInt{ qs.E } * W; // sqr( S ); the signs below are computed in Z[S]

    // ch[k] = 2 W^2 * ( vs[k].pt - sphereCenter ) = 2 W^2 ( vs[k].pt - vs[0].pt ) - W M - S w
    const Vector3i64 rel[4] = { {}, qs.u, qs.v, qs.q };
    std::array<SqrtVec, 4> ch;
    for ( int k = 0; k < 4; ++k )
        for ( int i = 0; i < 3; ++i )
            ch[k][i] = { 2 * W * W * rel[k][i] - W * BigInt{ qs.M[i] }, -BigInt{ qs.w[i] } };

    int order[4] = { 0, 1, 2, 3 };
    std::sort( std::begin( order ), std::end( order ), [&]( int l, int r ) { return vs[l].id < vs[r].id; } );

    for ( int idx : order )
    {
        if ( idx == 3 )
        {
            // perturbing the query point never moves the sphere: its largest (z) perturbation resolves
            // the tie, and an exactly tangential move exits the sphere
            return signOf( ch[3][2], ew ) < 0;
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
            return ( sz != 0 ? sG * sz * sN : sG * sN ) > 0;
        }
        // the first derivative vanishes for all perturbation directions of vs[idx]; the answer is given
        // by the curvature term (D-c)*(c-m) + sqr(rho), m = middle of two other triangle points,
        // rho = radius of the circle of centers
        SqrtVec chm;
        for ( int i = 0; i < 3; ++i )
            chm[i] = { W * BigInt{ qs.M[i] } - W * W * ( BigInt{ rel[q1][i] } + rel[q2][i] ), BigInt{ qs.w[i] } };
        const Vector3i64 qr{ vs[q1].pt - vs[q2].pt };
        auto f2 = dot( ch[3], chm, ew );
        f2.x += W * W * W * W * ( 4 * BigInt{ rSq } - BigInt{ dot( Vector3i128{ qr }, Vector3i128{ qr } ) } );
        if ( const int sF = signOf( f2, ew ) )
            return sF < 0;
        assert( false ); // possible only for an idle point, and those are skipped above
    }
    assert( false ); // the query point always resolves the tie
    return false;
}

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} //namespace MR
