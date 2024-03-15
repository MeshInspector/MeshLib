#pragma once
#include "MRMeshFwd.h"

namespace MR
{
/// \return All vertices on the positive side of the plane
/// \param pc Input point cloud that will be cut by the plane
/// \param plane Input plane to cut point cloud with
MRMESH_API VertBitSet findHalfSpacePoints( const PointCloud& pc, const Plane3f& plane );


struct DividePointCloudOptionalOutput
{
    /// optional out map from input points to output
    VertMap* outVmap{ nullptr };
    /// optional out other part of the point cloud
    PointCloud* otherPart{ nullptr };
    /// optional out map from input points to other part output
    VertMap* otherOutVmap{ nullptr };
};

/// This function cuts a point cloud with a plane, leaving only the part of mesh that lay in positive direction of normal
/// \return Point cloud object with vertices on the positive side of the plane
/// \param pc Input point cloud that will be cut by the plane
/// \param plane Input plane to cut point cloud with
/// \param optOut optional output of the function
MRMESH_API PointCloud divideWithPlane( const PointCloud& points, const Plane3f& plane, const DividePointCloudOptionalOutput& optOut = {} );
}
