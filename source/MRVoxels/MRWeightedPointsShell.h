#pragma once

#include "MRVoxelsFwd.h"
#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRClosestWeightedPoint.h"

namespace MR
{

struct WeightedPointsToDistanceVolumeParams
{
    DistanceVolumeParams vol;

    DistanceFromWeightedPointsComputeParams dist;

    /// if true, then the distance will get its sign from the normal of the closest point (positive values in the half space pointed by normal);
    /// initial distances must be unsigned then (e.g. all point weights are negative)
    bool signDistanceByNormal = false;
};

/// makes FunctionVolume representing minimal distance to weighted points
[[nodiscard]] MRVOXELS_API FunctionVolume weightedPointsToDistanceFunctionVolume( const PointCloud & cloud, const WeightedPointsToDistanceVolumeParams& params );

/// makes FunctionVolume representing minimal distance to mesh with weighted vertices
[[nodiscard]] MRVOXELS_API FunctionVolume weightedMeshToDistanceFunctionVolume( const Mesh & mesh, const WeightedPointsToDistanceVolumeParams& params );

struct WeightedPointsShellParameters
{
    /// build iso-surface of minimal distance to points corresponding to this value
    float offset = 0;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// parameters of distance finding
    DistanceFromWeightedPointsParams dist;

    /// if true, then the distance will get its sign from the normal of the closest point (positive values in the half space pointed by normal);
    /// initial distances must be unsigned then (e.g. all point weights are negative);
    /// true here allows one to construct one directional offset instead of bidirectional shell
    bool signDistanceByNormal = false;

    /// Progress callback
    ProgressCallback progress;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const WeightedPointsShellParameters& params );

/// consider a point cloud where each point has additive weight (taken from pointWeights and not from params),
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const VertScalars& pointWeights, const WeightedPointsShellParameters& params );

/// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> weightedMeshShell( const Mesh & mesh, const WeightedPointsShellParameters& params );

/// consider a mesh where each vertex has additive weight (taken from vertWeights and not from params), and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> weightedMeshShell( const Mesh & mesh, const VertScalars& vertWeights, const WeightedPointsShellParameters& params );

} //namespace MR
