#pragma once
#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRAABBTreePoints.h"

namespace MR
{
struct PointCloud;

/// \brief Makes consistent normals for valid points of given point cloud
/// \param avgNeighborhoodSize avg num of neighbors of each individual point
/// \ingroup PointCloudGroup
MRMESH_API VertCoords makeNormals( const PointCloud& pointCloud, 
                                   int avgNeighborhoodSize = 3 * AABBTreePoints::MaxNumPointsInLeaf );
}