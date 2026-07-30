#pragma once

#include "MRPrecisePredicates3.h"
#include "MRPch/MRBindingMacros.h"
#include <cmath>
#include <type_traits>

namespace MR
{

/// \ingroup MathGroup
/// \{

/// returns true if the point D is strictly inside the sphere of radius sqrt(rSq) passing via
/// points A, B, C, whose center is located on the positive side of plane ABC
/// (in the half-space pointed at by cross( b - a, c - a ) from the plane);
/// returns false in degenerate cases: A, B, C are collinear or rSq is smaller than the squared
/// circumradius of triangle ABC (no such sphere exists), and when D is exactly on the sphere;
/// cyclic permutations of (A, B, C) do not change the result, a swap of two of them selects the mirror sphere;
/// rSq must be given in the same integer grid units as the point coordinates
[[nodiscard]] MRMESH_API bool inSphere( const Vector3i & a, const Vector3i & b, const Vector3i & c,
    const Vector3i & d, std::int64_t rSq );

/// returns true if the point vs[3] is strictly inside the sphere of radius sqrt(rSq) passing via
/// points vs[0], vs[1], vs[2], with the center located on the positive side of plane vs[0]vs[1]vs[2]
/// (same convention as in the overload above);
/// resolves "vs[3] is exactly on the sphere" ties using simulation-of-simplicity: the points are
/// symbolically perturbed (larger perturbations of points with smaller ids; per point the z-coordinate
/// gets larger perturbation than y than x), and the first point whose perturbation moves vs[3] off
/// the sphere decides the answer;
/// known deviations from full simulation-of-simplicity semantics, all answered deterministically
/// for now (covered by sosInSphereDeviations test, to be resolved by ids in the following PRs):
///  * all three points of the triangle coincide: false now, while the perturbed triangle is not
///    degenerate and its sphere exists;
///  * two points of the triangle coincide and the third one is closer than the sphere's diameter:
///    false now, while the perturbed sphere may exist depending on the perturbation directions;
///  * rSq is exactly equal to the squared circumradius of a not-degenerate triangle: perturbations
///    change the sphere's existence, affecting both "vs[3] strictly inside" (true now) and
///    "vs[3] exactly on the sphere" (false now);
/// the remaining degenerate answers are exact under full simulation-of-simplicity, because no small
/// perturbation can create the sphere: distinct collinear triangle points, two coincident triangle
/// points with the third one at or beyond the sphere's diameter, rSq below the squared circumradius
[[nodiscard]] MRMESH_API bool inSphere( const std::array<PreciseVertCoords, 4> & vs, std::int64_t rSq );

/// returns true if the point d is strictly inside the sphere of radius sqrt(rSq) passing via
/// points a, b, c, whose center is located on the positive side of plane abc
/// (same convention as in the precise overloads above), all computed in floating-point:
/// the answers for the points near the sphere's surface are subject to rounding errors;
/// returns false in degenerate cases: collinear a, b, c or rSq below the squared circumradius
template <typename T>
[[nodiscard]] std::enable_if_t<std::is_floating_point_v<T>, bool> inSphere( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c,
    const Vector3<T> & d, T rSq )
{
    const auto u = b - a;
    const auto v = c - a;
    const auto w = cross( u, v ); // doubled normal of triangle abc
    const T ww = w.lengthSq();
    if ( ww <= 0 )
        return false; // a, b, c are collinear => no circle through them

    // circumcenter of triangle abc
    const auto o = a + ( u.lengthSq() * cross( v, w ) + v.lengthSq() * cross( w, u ) ) / ( 2 * ww );
    const T hh = rSq - ( a - o ).lengthSq(); // squared distance between the sphere's center and plane abc
    if ( hh < 0 )
        return false; // sqrt(rSq) is less than the circumradius of abc => no such sphere

    const auto center = o + std::sqrt( hh / ww ) * w;
    return ( d - center ).lengthSq() < rSq;
}

MR_BIND_TEMPLATE( bool inSphere( const Vector3f & a, const Vector3f & b, const Vector3f & c, const Vector3f & d, float rSq ) );
MR_BIND_TEMPLATE( bool inSphere( const Vector3d & a, const Vector3d & b, const Vector3d & c, const Vector3d & d, double rSq ) );

/// \}

} // namespace MR
