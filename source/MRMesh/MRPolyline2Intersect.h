#pragma once

#include "MRId.h"
#include "MREdgePoint.h"
#include <cfloat>

namespace MR
{
/// \addtogroup AABBTreeGroup
/// \{

/**
 * \brief detect if given point is inside polyline, by counting ray intersections
 * \param polyline input polyline
 * \param point input point
 */
[[nodiscard]] MRMESH_API bool isPointInsidePolyline( const Polyline2& polyline, const Vector2f& point );

/// \}

struct [[nodiscard]] PolylineIntersectionResult2
{
    /// intersection point in polyline
    EdgePoint edgePoint;
    /// stores the distance from ray origin to the intersection point in direction units
    float distanceAlongLine = 0;
};

/// Finds ray and polyline intersection in float-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// Finds the closest to ray origin intersection (or any intersection for better performance if \p !closestIntersect).
[[nodiscard]] MRMESH_API std::optional<PolylineIntersectionResult2> rayPolylineIntersect( const Polyline2& polyline, const Line2f& line,
    float rayStart = 0, float rayEnd = FLT_MAX, const IntersectionPrecomputes2<float>* prec = nullptr, bool closestIntersect = true );

/// Finds ray and polyline intersection in double-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// Finds the closest to ray origin intersection (or any intersection for better performance if \p !closestIntersect).
[[nodiscard]] MRMESH_API std::optional<PolylineIntersectionResult2> rayPolylineIntersect( const Polyline2& polyline, const Line2d& line,
    double rayStart = 0, double rayEnd = DBL_MAX, const IntersectionPrecomputes2<double>* prec = nullptr, bool closestIntersect = true );

} //namespace MR
