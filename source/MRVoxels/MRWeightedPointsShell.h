#pragma once

#include "MRVoxelsFwd.h"
#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRPointCloudVariadicOffset.h"

namespace MR
{

struct WeightedPointsToDistanceVolumeParams
{
    DistanceVolumeParams vol;

    DistanceFromWeightedPointsComputeParams dist;
};

/// makes FunctionVolume representing minimal distance to weighted points
[[nodiscard]] MRVOXELS_API FunctionVolume weightedPointsToDistanceFunctionVolume( const PointCloud & cloud, const WeightedPointsToDistanceVolumeParams& params );

struct WeightedPointsShellParameters
{
    /// build iso-surface of minimal distance to points corresponding to this value
    float offset = 0;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// parameters of distance finding
    DistanceFromWeightedPointsParams dist;

    /// Progress callback
    ProgressCallback progress;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const WeightedPointsShellParameters& params );

} //namespace MR
