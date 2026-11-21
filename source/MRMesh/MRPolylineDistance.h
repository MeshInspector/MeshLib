#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

/**
 * \brief returns the maximum of the squared distances from each vertex of B-polyline to A-polyline
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
[[nodiscard]] MRMESH_API float findMaxDistanceSqOneWay( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief returns the squared Hausdorff distance between two polylines measured in their vertices, that is
          the maximum of squared distances from each vertex of one polyline to the other polyline (in both directions)
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
[[nodiscard]] MRMESH_API float findMaxDistanceSq( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

} // namespace MR
