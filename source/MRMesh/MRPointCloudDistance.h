#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

/**
 * \brief returns the maximum of the squared distances from each B-point to A-cloud
 * \param rigidB2A rigid transformation from B-cloud space to A-cloud space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
[[nodiscard]] MRMESH_API float findMaxDistanceSqOneWay( const PointCloud& a, const PointCloud& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief returns the squared Hausdorff distance between two point clouds, that is
          the maximum of squared distances from each point to the other cloud (in both directions)
 * \param rigidB2A rigid transformation from B-cloud space to A-cloud space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
 */
[[nodiscard]] MRMESH_API float findMaxDistanceSq( const PointCloud& a, const PointCloud& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

} // namespace MR
