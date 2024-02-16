#pragma once
#include "MRMeshFwd.h"

namespace MR
{
/// \return All vertices on the positive side of the plane
/// \param pc Input point cloud that will be cut by the plane
/// \param plane Input plane to cut point cloud with
MRMESH_API VertBitSet findHalfSpacePoints( const PointCloud& pc, const Plane3f& plane );

/// This function cuts a point cloud with a plane, leaving only the part of mesh that lay in positive direction of normal
/// \return Point cloud object with vertices on the positive side of the plane
/// \param pc Input point cloud that will be cut by the plane
/// \param plane Input plane to cut point cloud with
/// \param otherPart Optional return other part of the point cloud
MRMESH_API PointCloud divideWithPlane( const PointCloud& points, const Plane3f& plane, PointCloud* otherPart = nullptr );
}
