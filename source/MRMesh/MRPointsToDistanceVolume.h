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

/// given
/// \param cloud      a point cloud
/// \param colors     colors of each point in the cloud
/// \param tgtPoints  some target points
/// \param tgtVerts   mask of valid target points
/// \param sigma      the distance of highest influence of a point
/// \param cb         progress callback
/// computes the colors in valid target points by averaging the colors from the point cloud
[[nodiscard]] MRMESH_API Expected<VertColors> calcAvgColors( const PointCloud & cloud, const VertColors & colors,
    const VertCoords & tgtPoints, const VertBitSet & tgtVerts, float sigma, const ProgressCallback & cb = {} );

} //namespace MR
