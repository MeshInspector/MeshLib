#pragma once
#include "MRMeshFwd.h"

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

}