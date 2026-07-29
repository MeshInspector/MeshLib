#pragma once

#include "MRPrecisePredicates3.h"

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

/// \}

} // namespace MR
