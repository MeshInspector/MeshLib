#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRExpected.h"

namespace MR
{

struct PointsToDistanceVolumeParams : DistanceVolumeParams
{
    /// distance of influence of a point, beyond that radius the influence is zero
    float influenceRadius = 0;

    /// signed distance is truncated (reaches its maximum or minimum) at this distance
    float truncationRadius = 0;

    /// minimum number of points located within influenceRadius for a voxel to get a value
    int minInfluencePoints = 1;
};

/// makes SimpleVolume filled with signed distances to points with normals
[[nodiscard]] MRMESH_API Expected<SimpleVolume> pointsToDistanceVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params );

} //namespace MR
