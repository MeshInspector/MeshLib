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

/**
 * \brief finds the point of B-polyline (inside its segments as well, not only in vertices) most distant from A-polyline;
 *        the distance to it is the one-sided Hausdorff distance from B to A
 * \details the maximum can be reached inside a segment of B, where several segments of A are equally close,
 *          so the segments of B are halved there till the given tolerance is reached
 * \return the found point in b, its projection on A-polyline in a, and the squared distance between them
 * \param tolerance positive value, the returned distance (not squared) is never above the exact one, and is below it by at most this value
 *        (both up to rounding errors in floats); the smaller the value, the longer the computation;
 *        if it is not positive then only the vertices are considered
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and no valid point in a
 */
[[nodiscard]] MRMESH_API PolylinePolylineDistanceResult findHausdorffDistanceOneWay( const Polyline3& a, const Polyline3& b, float tolerance, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

/**
 * \brief finds the point of one polyline (not only in vertices) most distant from the other polyline (in both directions);
 *        the distance to it is the Hausdorff distance between two polylines
 * \return the found point in a or in b, its projection on the other polyline in the remaining field, and the squared distance between them
 * \param tolerance positive value, the returned distance (not squared) is never above the exact one, and is below it by at most this value
 *        (both up to rounding errors in floats); the smaller the value, the longer the computation;
 *        if it is not positive then only the vertices are considered
 * \param rigidB2A rigid transformation from B-polyline space to A-polyline space, nullptr considered as identity transformation
 * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq and one valid point only
 */
[[nodiscard]] MRMESH_API PolylinePolylineDistanceResult findHausdorffDistance( const Polyline3& a, const Polyline3& b, float tolerance, const AffineXf3f* rigidB2A = nullptr, float maxDistanceSq = FLT_MAX );

} // namespace MR
