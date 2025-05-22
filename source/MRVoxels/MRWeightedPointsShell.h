#pragma once

#include "MRVoxelsFwd.h"
#include "MRDistanceVolumeParams.h"
#include "MRMarchingCubes.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRClosestWeightedPoint.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRBox.h"
#include "MRPch/MRBindingMacros.h"

namespace MR::WeightedShell
{

struct DistanceVolumeCreationParams
{
    DistanceVolumeParams vol;

    DistanceFromWeightedPointsComputeParams dist;
};

/// makes FunctionVolume representing minimal distance to weighted points
[[nodiscard]] MRVOXELS_API FunctionVolume pointsToDistanceVolume( const PointCloud & cloud, const DistanceVolumeCreationParams& params );

/// makes FunctionVolume representing minimal distance to mesh with weighted vertices
[[nodiscard]] MRVOXELS_API FunctionVolume meshToDistanceVolume( const Mesh & mesh, const DistanceVolumeCreationParams& params );
using meshToDistanceVolumeT = std::function<FunctionVolume(const Mesh&, const DistanceVolumeCreationParams&)>;

struct ParametersBase
{
    /// build iso-surface of minimal distance to points corresponding to this value
    float offset = 0;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// number of voxels to compute near the offset (should be left default unless used for debugging)
    float numLayers = 1.001f;

    /// Progress callback
    ProgressCallback progress;


    // To allow passing Python lambdas into `dist.pointWeight`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

struct ParametersMetric : ParametersBase
{
    /// parameters of distance finding
    DistanceFromWeightedPointsParams dist;
};

struct ParametersRegions : ParametersBase
{
    struct Region
    {
        VertBitSet verts;
        float weight = 0.f;
    };

    /// list of regions (overlappings are allowed) with corresponding offsets
    /// the additional offset in overlaps is set to the average of the regions
    std::vector<Region> regions;

    /// interpolation distance between the weights of the regions
    /// determines the sharpness of transitions between different regions
    float interpolationDist = 0;

    /// if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
    /// if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
    bool bidirectionalMode = true;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
// MR_BIND_IGNORE to hide this function in Python API because calling Python's Lambda will be extremely slow anyway
[[nodiscard]] MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> pointsShell( const PointCloud & cloud, const ParametersMetric& params );

/// consider a point cloud where each point has additive weight (taken from pointWeights and not from params),
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> pointsShell( const PointCloud & cloud, const VertScalars& pointWeights, const ParametersMetric& params );

/// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
// MR_BIND_IGNORE to hide this function in Python API because calling Python's Lambda will be extremely slow anyway
[[nodiscard]] MR_BIND_IGNORE MRVOXELS_API Expected<Mesh> meshShell( const Mesh & mesh, const ParametersMetric& params );

/// consider a mesh where each vertex has additive weight (taken from vertWeights and not from params), and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
[[nodiscard]] MRVOXELS_API Expected<Mesh> meshShell( const Mesh & mesh, const VertScalars& vertWeights, const ParametersMetric& params );

/// interpolate set of regions and assign weight to each vertex of the mesh
MRVOXELS_API VertScalars calculateShellWeightsFromRegions(
    const Mesh& mesh, const std::vector<ParametersRegions::Region>& regions, float interpolationDist );

/// this overload supports linear interpolation between the regions with different weight
[[nodiscard]] MRVOXELS_API Expected<Mesh> meshShell( const Mesh& mesh, const ParametersRegions& params );

/// this overload allows to control how distance volume is build during the offset
[[nodiscard]] MRVOXELS_API Expected<Mesh> meshShell( const Mesh& mesh, const ParametersRegions& params, meshToDistanceVolumeT volumeBuilder );

} //namespace MR::WeightedShell
