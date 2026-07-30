#pragma once

#include "MRPrecisePredicates3.h"
#include "MRPch/MRBindingMacros.h"
#include <type_traits>

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
    /// the sphere exists, and the point D is on it or outside
    Outside,
    /// the sphere exists, and the point D is strictly inside
    Inside
};

/// checks whether the point D is strictly inside the sphere of radius sqrt(rSq) passing via
/// points A, B, C, whose center is located on the positive side of plane ABC
/// (in the half-space pointed at by cross( b - a, c - a ) from the plane);
/// returns Outside when D is exactly on the sphere;
/// cyclic permutations of (A, B, C) do not change the result, a swap of two of them selects the mirror sphere;
/// rSq must be given in the same integer grid units as the point coordinates
[[nodiscard]] MRMESH_API InSphereResult inSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c,
    const Vector3i & d, std::int64_t rSq );

/// checks whether the point vs[3] is strictly inside the sphere of radius sqrt(rSq) passing via
/// points vs[0], vs[1], vs[2], with the center located on the positive side of plane vs[0]vs[1]vs[2]
/// (same convention as in the overload above);
/// resolves "vs[3] is exactly on the sphere" ties into Inside or Outside using simulation-of-simplicity:
/// the points are symbolically perturbed (larger perturbations of points with smaller ids; per point the
/// z-coordinate gets larger perturbation than y than x), and the first point whose perturbation moves
/// vs[3] off the sphere decides the answer;
/// known deviations from full simulation-of-simplicity semantics, all answered deterministically
/// for now (covered by sosInSphereDeviations test, to be resolved by ids in the following PRs):
///  * all three points of the triangle coincide: NoSphere now, while the perturbed triangle is not
///    degenerate and its sphere exists;
///  * two points of the triangle coincide and the third one is closer than the sphere's diameter:
///    NoSphere now, while the perturbed sphere may exist depending on the perturbation directions;
///  * rSq is exactly equal to the squared circumradius of a not-degenerate triangle: perturbations
///    change the sphere's existence, affecting both "vs[3] strictly inside" (Inside now) and
///    "vs[3] exactly on the sphere" (Outside now);
/// the remaining degenerate NoSphere answers are exact under full simulation-of-simplicity, because
/// no small perturbation can create the sphere: distinct collinear triangle points, two coincident
/// triangle points with the third one at or beyond the sphere's diameter, rSq below the squared circumradius
[[nodiscard]] MRMESH_API InSphereResult inSphere( const std::array<PreciseVertCoords, 4> & vs, std::int64_t rSq );

/// checks whether the point d is strictly inside the sphere of radius sqrt(rSq) passing via
/// points a, b, c, whose center is located on the positive side of plane abc
/// (same convention and case analysis as in the precise overloads above), computed in floating-point:
/// the answers for the points near the sphere's surface are subject to rounding errors;
/// the products inside have degree 16 in coordinates, so to avoid overflows any difference of two
/// given points' coordinates as well as sqrt(rSq) must be below ~100 for T=float and ~1e18 for T=double
template <typename T>
[[nodiscard]] std::enable_if_t<std::is_floating_point_v<T>, InSphereResult> inSphere( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, T rSq )
{
    const auto u = b - a;
    const auto v = c - a;
    const auto q = d - a;

    const auto w = cross( u, v ); // doubled normal of triangle abc
    const T W = w.lengthSq();
    if ( W <= 0 )
        return InSphereResult::NoSphere; // a, b, c are collinear => no circle through them

    const auto M = u.lengthSq() * cross( v, w ) + v.lengthSq() * cross( w, u ); // 2 * W * ( circumcenter(abc) - a )
    const T E = 4 * rSq * W * W - M.lengthSq(); // sqr( 2 * h * W ), h = distance from plane abc to the sphere's center
    if ( E < 0 )
        return InSphereResult::NoSphere; // sqrt(rSq) is less than the circumradius of abc => no such sphere

    const T A = W * q.lengthSq() - dot( q, M ); // W * ( |d - circumcenter(abc)|^2 - sqr( circumradius ) )
    const T t = dot( q, w ); // |w| * signedDistance( d, plane abc )

    // d is strictly inside the sphere <=> A * |w| < sqrt( E ) * t
    if ( A < 0 && t >= 0 )
        return InSphereResult::Inside;
    if ( A >= 0 && t <= 0 )
        return InSphereResult::Outside;
    const T lhs = A * A * W;
    const T rhs = E * t * t;
    if ( A < 0 )
        return lhs > rhs ? InSphereResult::Inside : InSphereResult::Outside;
    return lhs < rhs ? InSphereResult::Inside : InSphereResult::Outside;
}

MR_BIND_TEMPLATE( InSphereResult inSphere( const Vector3f & a, const Vector3f & b, const Vector3f & c, const Vector3f & d, float rSq ) );
MR_BIND_TEMPLATE( InSphereResult inSphere( const Vector3d & a, const Vector3d & b, const Vector3d & c, const Vector3d & d, double rSq ) );

/// \}

} // namespace MR
