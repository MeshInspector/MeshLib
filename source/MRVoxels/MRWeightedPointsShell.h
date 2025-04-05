#pragma once

#include "MRVoxelsFwd.h"
#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRClosestWeightedPoint.h"
#include "MRMesh/MRBitSet.h"
#include "MRPch/MRBindingMacros.h"

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

struct WeightedPointsShellParametersBase
{
    /// build iso-surface of minimal distance to points corresponding to this value
    float offset = 0;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// if true, then the distance will get its sign from the normal of the closest point (positive values in the half space pointed by normal);
    /// initial distances must be unsigned then (e.g. all point weights are negative);
    /// true here allows one to construct one directional offset instead of bidirectional shell
    bool signDistanceByNormal = false;

    /// Progress callback
    ProgressCallback progress;


    // To allow passing Python lambdas into `dist.pointWeight`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

struct WeightedPointsShellParametersMetric : WeightedPointsShellParametersBase
{
    /// parameters of distance finding
    DistanceFromWeightedPointsParams dist;
};

struct WeightedPointsShellParametersRegions : WeightedPointsShellParametersBase
{
    struct PartialVertScalars
    {
        VertBitSet verts;
        float value = 0.f;
    };

    /// list of regions (overlappings are allowed) with corresponding offsets
    std::vector<PartialVertScalars> regions;

    /// interpolation factor between the weights of the regions
    float interpolationDist = 0;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
// MR_BIND_IGNORE to hide this function in Python API because calling Python's Lambda will be extremely slow anyway
[[nodiscard]] MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const WeightedPointsShellParametersMetric& params );

/// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
// MR_BIND_IGNORE to hide this function in Python API because calling Python's Lambda will be extremely slow anyway
[[nodiscard]] MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> weightedMeshShell( const Mesh & mesh, const WeightedPointsShellParametersMetric& params );

/// this overload supports linear interpolation between the regions with different weight
[[nodiscard]] MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> weightedMeshShell( const Mesh& mesh, const WeightedPointsShellParametersRegions& params );

} //namespace MR
