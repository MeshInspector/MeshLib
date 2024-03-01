#pragma once

#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct PointsToMeshParameters
{
    /// it the distance of highest influence of a point;
    /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
    float sigma = 1;

    /// minimum number of points located within influenceRadius for a voxel to get a value
    int minInfluencePoints = 1;

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
