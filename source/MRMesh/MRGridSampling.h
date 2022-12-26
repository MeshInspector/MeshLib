#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

// performs sampling of mesh vertices;
// subdivides mesh bounding box on voxels of approximately given size and returns at most one vertex per voxel
MRMESH_API VertBitSet verticesGridSampling( const MeshPart& mp, float voxelSize, ProgressCallback cb = {} );
// the same for point cloud
MRMESH_API VertBitSet pointGridSampling( const PointCloud& cloud, float voxelSize, ProgressCallback cb = {} );

} //namespace MR
