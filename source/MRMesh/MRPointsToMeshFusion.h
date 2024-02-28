#pragma once

#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct PointsToMeshParameters
{
    /// distance of influence of a point, beyond that radius the influence is zero
    float influenceRadius = 0;

    /// signed distance is truncated (reaches its maximum or minimum) at this distance
    float truncationRadius = 0;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// Progress callback
    ProgressCallback progress;
};

/// makes mesh from points with normals by constructing intermediate volume with signed distances
/// and then using marching cubes algorithm to extract the surface from there
[[nodiscard]] MRMESH_API Expected<Mesh> pointsToMeshFusion( const PointCloud & cloud, const PointsToMeshParameters& params );

} //namespace MR
