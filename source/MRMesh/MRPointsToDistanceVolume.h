#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRExpected.h"

namespace MR
{

struct PointsToDistanceVolumeParams : DistanceVolumeParams
{
    /// it the distance of highest influence of a point;
    /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
    float sigma = 1;

    /// minimum sum of influence weights from surrounding points for a voxel to get a value, meaning that there shall be at least this number of points in close proximity
    float minWeight = 1;
};

/// makes SimpleVolume filled with signed distances to points with normals
[[nodiscard]] MRMESH_API Expected<SimpleVolume> pointsToDistanceVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params );

} //namespace MR
