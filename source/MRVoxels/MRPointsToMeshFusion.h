#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"

#include <memory>

namespace MR
{

class IComputePointsToDistanceVolume;

struct PointsToMeshParameters
{
    /// it the distance of highest influence of a point;
    /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
    float sigma = 1;

    /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
    float minWeight = 1;

    /// coefficient used for weight calculation: e^(dist^2 * -invSigmaModifier * sigma^-2)
    /// values: (0;inf)
    float invSigmaModifier = 0.5f;

    /// changes the way point angle affects weight, by default it is linearly increasing with dot product
    /// if enabled - increasing as dot product^(0.5) (with respect to its sign)
    bool sqrtAngleWeight{ false };

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// optional input: colors of input points
    const VertColors * ptColors = nullptr;

    /// optional output: averaged colors of mesh vertices
    VertColors * vColors = nullptr;

    /// Progress callback
    ProgressCallback progress;

    /// builds the intermediate volume, e.g. MR::Cuda::ComputePointsToDistanceVolume to build it on GPU;
    /// if it is not set or cannot process this input, MR::ComputePointsToDistanceVolume is used
    std::shared_ptr<IComputePointsToDistanceVolume> computeVolume;
};

/// makes mesh from points with normals by constructing intermediate volume with signed distances
/// and then using marching cubes algorithm to extract the surface from there
[[nodiscard]] MRVOXELS_API Expected<Mesh> pointsToMeshFusion( const PointCloud & cloud, const PointsToMeshParameters& params );

} //namespace MR
