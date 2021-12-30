#pragma once
#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRAABBTreePoints.h"

namespace MR
{
struct PointCloud;

// Makes consistent normals for valid points of given point cloud
// avgNeighborhoodSize avg num of neighbors of each individual point
MRMESH_API VertCoords makeNormals( const PointCloud& pointCloud, 
                                   int avgNeighborhoodSize = 3 * AABBTreePoints::MaxNumPointsInLeaf );
}