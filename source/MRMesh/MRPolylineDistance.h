#pragma once

#include "MREdgePoint.h"
#include <cfloat>

namespace MR
{

struct PolylinePolylineDistanceResult
{
    /// two points: from polylines A and B respectively, each relative to the topology of its own polyline
    EdgePoint a, b;
    /// squared distance between a and b
    float distSq = 0;
};

/**
 * \brief finds the vertex of B-polyline most distant from A-polyline
 * \return the found vertex in b, its projection on A-polyline in a, and the squared distance between them
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and no valid point in a
 */
[[nodiscard]] MRMESH_API PolylinePolylineDistanceResult findMaxDistanceOneWay( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief finds the vertex of one polyline most distant from the other polyline (in both directions),
 *        the distance to it is the Hausdorff distance between two polylines measured in their vertices
 * \return the found vertex in a or in b, its projection on the other polyline in the remaining field, and the squared distance between them
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and one valid point only
 */
[[nodiscard]] MRMESH_API PolylinePolylineDistanceResult findMaxDistance( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

} // namespace MR
