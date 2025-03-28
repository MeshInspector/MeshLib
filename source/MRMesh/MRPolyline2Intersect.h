#pragma once

#include "MRId.h"
#include "MREdgePoint.h"
#include "MREnums.h"
#include <cfloat>
#include <optional>

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

/// this callback is envoked for each encountered ray-polyline intersection;
/// if it returns Processing::Stop, then the search immediately terminates;
/// the callback can reduce rayEnd affecting the following search, but it shall not increase rayEnd
template<typename T>
using PolylineIntersectionCallback2 = std::function<Processing(const EdgePoint & polylinePoint, T rayPos, T & rayEnd)>;
using PolylineIntersectionCallback2f = PolylineIntersectionCallback2<float>;
using PolylineIntersectionCallback2d = PolylineIntersectionCallback2<double>;

/// Intersects 2D ray and polyline in single-precision.
/// Reports all intersections via given callback with the tendency to do it from ray start to ray end, but without guarantee of exact order.
MRMESH_API void rayPolylineIntersectAll( const Polyline2& polyline, const Line2f& line, const PolylineIntersectionCallback2f& callback,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes2<float>* prec = nullptr );

/// Intersects 2D ray and polyline in double-precision.
/// Reports all intersections via given callback with the tendency to do it from ray start to ray end, but without guarantee of exact order.
MRMESH_API void rayPolylineIntersectAll( const Polyline2& polyline, const Line2d& line, const PolylineIntersectionCallback2d& callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes2<double>* prec = nullptr );

} //namespace MR
