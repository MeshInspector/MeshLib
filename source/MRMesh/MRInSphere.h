#pragma once

#include "MRPrecisePredicates3.h"
#include "MRFastInt.h"
#include "MRPch/MRBindingMacros.h"
#include <array>
#include <cassert>
#include <type_traits>
#include <utility>

namespace MR
{

/// \ingroup MathGroup
/// \{

/// the result of inSphere predicates
enum class InSphereResult
{
    /// the sphere is not defined: the points A, B, C are collinear or coincident,
    /// or rSq is less than the squared circumradius of triangle ABC
    NoSphere,
    /// the sphere exists, and the point D is strictly outside
    Outside,
    /// the sphere exists, and the point D is exactly on it;
    /// never returned by the simulation-of-simplicity overload, which resolves
    /// such ties into Inside or Outside
    OnSphere,
    /// the sphere exists, and the point D is strictly inside
    Inside
};

/// checks whether the point D is strictly inside the sphere of radius sqrt(rSq) passing via
/// points A, B, C, whose center is located on the positive side of plane ABC
/// (in the half-space pointed at by cross( b - a, c - a ) from the plane);
/// returns OnSphere when D is exactly on the sphere;
/// cyclic permutations of (A, B, C) do not change the result, a swap of two of them selects the mirror sphere;
/// rSq must be given in the same integer grid units as the point coordinates
[[nodiscard]] MRMESH_API InSphereResult inSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c,
    const Vector3i & d, std::int64_t rSq );

/// checks whether the point vs[3] is strictly inside the sphere of radius sqrt(rSq) passing via
/// points vs[0], vs[1], vs[2], with the center located on the positive side of plane vs[0]vs[1]vs[2]
/// (same convention as in the overload above);
/// full simulation-of-simplicity semantics: the answer equals the answer of the plain predicate
/// on the symbolically perturbed points, where the point with rank r among the four (by ascending
/// ids) receives +eps^(9*L^r) to x, +eps^(3*L^r) to y, +eps^(L^r) to z with L = 128, eps -> +0
/// (z gets the largest perturbation as in getPointDegrees, but the ladder between the ranks is 128:
/// it exceeds the maximal degree-weight of one point in the predicate polynomials, which makes the
/// resolution independent of whether the ranks are computed among 3 or among all 4 points);
/// consequently:
///  * "vs[3] exactly on the sphere" resolves into Inside or Outside by the ids;
///  * rSq exactly equal to the squared circumradius: the perturbation decides the sphere's
///    existence, so the answer can be NoSphere even for vs[3] strictly inside;
///  * three coincident or distinct collinear triangle points always give NoSphere: the perturbed
///    triangle is needle-like (the perturbation magnitudes differ vastly) and its circumradius
///    diverges as eps -> +0;
///  * two coincident triangle points: the sphere exists iff 4*rSq*(Vx^2+Vy^2) > |V|^4 at the
///    leading order (the pair separates along z), V = the third point minus the pair
[[nodiscard]] MRMESH_API InSphereResult inSphere( const std::array<PreciseVertCoords, 4> & vs, std::int64_t rSq );

/// accelerates testing of many query points against one sphere given by the same three points
/// and radius, pre-computing all the point-independent quantities once;
/// the primary template works for floating-point T with the same case analysis as the precise
/// specialization for int below, so the answers for the query points near the sphere's surface
/// are subject to rounding errors, and OnSphere is returned only on the exact equality in the
/// comparisons; the products inside have degree 16 in coordinates, so to avoid overflows any
/// difference of two given points' coordinates as well as sqrt(rSq) must be below ~100 for T=float
/// and ~1e18 for T=double;
/// the one-call inSphere functions are implemented via this class
template <typename T>
class InSphereTester
{
    static_assert( std::is_floating_point_v<T> );
public:
    /// prepares the tester for the sphere of radius sqrt(rSq) passing via points a, b, c, with the
    /// center located on the positive side of plane abc (in the half-space pointed at by
    /// cross( b - a, c - a ) from the plane);
    /// returns false if no such sphere exists: the points are collinear or coincident,
    /// or rSq is less than the squared circumradius of triangle abc
    bool reset( const Vector3<T> & va, const Vector3<T> & vb, const Vector3<T> & vc, T sqRadius )
    {
        a = va;
        rSq = sqRadius;
        E = -1;
        u = vb - va;
        v = vc - va;

        // no sphere of radius sqrt(rSq) can pass via two points more than the diameter apart;
        // strictly greater: a side exactly equal to the diameter can lie on the sphere
        const T uu = u.lengthSq();
        const T vv = v.lengthSq();
        if ( uu > 4 * rSq || vv > 4 * rSq || ( v - u ).lengthSq() > 4 * rSq )
            return false;

        w = cross( u, v );
        W = w.lengthSq();
        if ( W <= 0 )
            return false; // a, b, c are collinear => no circle through them

        // 2 * W * ( circumcenter(abc) - a ), expanded as in circumcircleCenter
        const T uv = dot( u, v );
        M = ( vv * ( uu - uv ) ) * u + ( uu * ( vv - uv ) ) * v;

        // negative: sqrt(rSq) is less than the circumradius of the triangle => no such sphere
        E = 4 * rSq * W * W - M.lengthSq();
        return E >= 0;
    }

    /// swaps the points b and c, which selects the mirror sphere with the center on the other side
    /// of plane abc, and gives exactly the state of reset( a, c, b, rSq ) without recomputing anything:
    /// only the normal w changes there, and W, M, E are symmetric in b and c;
    /// shall be called only after reset() returned true
    void flip()
    {
        assert( E >= 0 ); // the last reset() must have returned true
        std::swap( u, v );
        w = -w;
    }

    /// returns the position of the point d relative to the sphere (never NoSphere);
    /// shall be called only after reset() returned true
    [[nodiscard]] InSphereResult operator()( const Vector3<T> & d ) const
    {
        assert( E >= 0 ); // the last reset() must have returned true
        const auto q = d - a;

        // d farther than the diameter from a point on the sphere is strictly outside
        const T qq = q.lengthSq();
        if ( qq > 4 * rSq )
            return InSphereResult::Outside;

        const T A = W * qq - dot( q, M ); // W * ( |d - circumcenter|^2 - sqr( circumradius ) )
        const T t = dot( q, w ); // |w| * signedDistance( d, plane of the triangle )

        // d is strictly inside the sphere <=> A * |w| < sqrt( E ) * t
        if ( A < 0 && t >= 0 )
            return InSphereResult::Inside;
        if ( A >= 0 && t <= 0 )
            return ( A == 0 && ( t == 0 || E == 0 ) ) ? InSphereResult::OnSphere : InSphereResult::Outside;
        const T lhs = A * A * W;
        const T rhs = E * t * t;
        if ( lhs == rhs )
            return InSphereResult::OnSphere;
        return ( A < 0 ) == ( lhs > rhs ) ? InSphereResult::Inside : InSphereResult::Outside;
    }

private:
    Vector3<T> a;    ///< the first sphere point
    Vector3<T> u, v; ///< b - a, c - a
    Vector3<T> w;    ///< doubled normal of triangle abc
    T W = 0;         ///< |w|^2
    Vector3<T> M;    ///< 2 * |w|^2 * ( circumcenter(abc) - a )
    T E = -1;        ///< sqr( 2 * h * |w|^2 ), h = distance from plane abc to the sphere's center
    T rSq = 0;       ///< the squared radius of the sphere
};


/// the specialization implementing the precise integer predicate, exact for any input;
/// rSq must be given in the same integer grid units as the point coordinates
template <>
class InSphereTester<int>
{
public:
    /// prepares the tester for the sphere of radius sqrt(rSq) passing via points a, b, c, with the
    /// center located on the positive side of plane abc (in the half-space pointed at by
    /// cross( b - a, c - a ) from the plane);
    /// returns false if no such sphere exists: the points are collinear or coincident,
    /// or rSq is less than the squared circumradius of triangle abc
    MRMESH_API bool reset( const Vector3i & a, const Vector3i & b, const Vector3i & c, std::int64_t rSq );

    /// swaps the points b and c, which selects the mirror sphere with the center on the other side
    /// of plane abc, and gives exactly the state of reset( a, c, b, rSq ) without recomputing anything:
    /// only the normal w changes there, and W, M, E are symmetric in b and c;
    /// shall be called only after reset() returned true
    void flip()
    {
        assert( E >= 0 ); // the last reset() must have returned true
        std::swap( u, v );
        w = -w;
    }

    /// returns the position of the point d relative to the sphere (never NoSphere);
    /// shall be called only after reset() returned true
    [[nodiscard]] MRMESH_API InSphereResult operator()( const Vector3i & d ) const;

protected:
    Vector3i a;           ///< the first sphere point
    Vector3i64 u, v;      ///< b - a, c - a
    Vector3i64 w;         ///< doubled normal of triangle abc, <= 2^63
    FastInt<192> W;       ///< |w|^2, <= 2^128
    std::array<FastInt<192>, 3> M; ///< 2 * |w|^2 * ( circumcenter(abc) - a ), <= 2^161
    FastInt<384> E = -1;  ///< sqr( 2 * h * |w|^2 ), h = distance from plane abc to the sphere's center, <= 2^322
    std::int64_t rSq = 0; ///< the squared radius of the sphere
};

using InSphereTesterf = InSphereTester<float>;
using InSphereTesterd = InSphereTester<double>;
using InSphereTesteri = InSphereTester<int>;

/// the precise tester with simulation-of-simplicity resolution of "exactly on the sphere" ties,
/// which reuses all the geometric machinery of InSphereTester<int> and adds only the vertex ids
class InSphereTesterSoS : public InSphereTester<int>
{
public:
    /// prepares the tester for the sphere of radius sqrt(rSq) passing via points a.pt, b.pt, c.pt,
    /// with the center located on the positive side of their plane (in the half-space pointed at by
    /// cross( b.pt - a.pt, c.pt - a.pt ) from the plane), remembering the ids for the queries;
    /// returns false if no such sphere exists for the symbolically perturbed points (see the
    /// simulation-of-simplicity inSphere above): in particular, coincident and collinear triangle
    /// points are resolved by the perturbation here, and rSq exactly equal to the squared
    /// circumradius gives false whenever the perturbation of these ids breaks the existence;
    /// this hides the id-less reset of the base class, which would leave stale ids
    MRMESH_API bool reset( const PreciseVertCoords & a, const PreciseVertCoords & b, const PreciseVertCoords & c, std::int64_t rSq );

    /// swaps the points b and c together with their ids, which selects the mirror sphere as in the
    /// base class, and gives exactly the state of reset( a, c, b, rSq );
    /// this hides the id-less flip of the base class, which would leave stale ids
    void flip()
    {
        if ( degenerateTriangle_ )
            std::swap( u, v ); // w stays exactly zero for a degenerate triangle
        else
            InSphereTester<int>::flip();
        std::swap( vb_, vc_ );
    }

    /// returns the position of the point d.pt relative to the sphere, resolving "exactly on the
    /// sphere" ties into Inside or Outside by simulation-of-simplicity as described at the inSphere
    /// overload above, using the ids of d and of the points given in reset()
    /// (never returns OnSphere or NoSphere);
    /// shall be called only after reset() returned true, and all four ids must be distinct
    [[nodiscard]] MRMESH_API InSphereResult operator()( const PreciseVertCoords & d ) const;

private:
    VertId va_, vb_, vc_; ///< the ids of the sphere points given in reset()
    bool degenerateTriangle_ = false; ///< reset() got W == 0, and the queries go via the full symbolic evaluation
};

/// checks whether the point d is strictly inside the sphere of radius sqrt(rSq) passing via
/// points a, b, c, whose center is located on the positive side of plane abc, in floating-point;
/// see the comment on InSphereTester above for the limitations
template <typename T>
[[nodiscard]] std::enable_if_t<std::is_floating_point_v<T>, InSphereResult> inSphere( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, T rSq )
{
    InSphereTester<T> tester;
    if ( !tester.reset( a, b, c, rSq ) )
        return InSphereResult::NoSphere;
    return tester( d );
}

MR_BIND_TEMPLATE( InSphereResult inSphere( const Vector3f & a, const Vector3f & b, const Vector3f & c, const Vector3f & d, float rSq ) );
MR_BIND_TEMPLATE( InSphereResult inSphere( const Vector3d & a, const Vector3d & b, const Vector3d & c, const Vector3d & d, double rSq ) );

/// \}

} // namespace MR
