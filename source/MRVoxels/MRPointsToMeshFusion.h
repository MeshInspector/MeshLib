#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"

namespace MR
{

struct PointsToDistanceVolumeParams;

struct PointsToMeshParameters
{
    /// it the distance of highest influence of a point;
    /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
    float sigma = 1;

    /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
    float minWeight = 1;

    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// optional input: colors of input points
    const VertColors * ptColors = nullptr;

    /// optional output: averaged colors of mesh vertices
    VertColors * vColors = nullptr;

    /// Progress callback
    ProgressCallback progress;

    /// Callback for volume creation. If null - volume will be created with memory efficient pointsToDistanceFunctionVolume function
    std::function<Expected<SimpleVolumeMinMax>( const PointCloud& cloud, const PointsToDistanceVolumeParams& params )> createVolumeCallback;
};

/// makes mesh from points with normals by constructing intermediate volume with signed distances
/// and then using marching cubes algorithm to extract the surface from there
[[nodiscard]] MRVOXELS_API Expected<Mesh> pointsToMeshFusion( const PointCloud & cloud, const PointsToMeshParameters& params );

} //namespace MR
