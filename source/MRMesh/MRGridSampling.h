#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// performs sampling of mesh vertices;
/// subdivides mesh bounding box on voxels of approximately given size and returns at most one vertex per voxel;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<VertBitSet> verticesGridSampling( const MeshPart& mp, float voxelSize, const ProgressCallback & cb = {} );

/// performs sampling of cloud points;
/// subdivides point cloud bounding box on voxels of approximately given size and returns at most one point per voxel;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<VertBitSet> pointGridSampling( const PointCloud& cloud, float voxelSize, const ProgressCallback & cb = {} );

} //namespace MR
