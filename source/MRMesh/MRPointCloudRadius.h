#pragma once
#include "MRMeshFwd.h"

namespace MR
{
struct PointCloud;

/// \brief Finds radius of ball that avg points in the radius is close to avgPoints parameter
/// \ingroup PointCloudGroup
MRMESH_API float findAvgPointsRadius( const PointCloud& pointCloud, int avgPoints );

/// expands the region on given euclidian distance. returns false if callback also returns false
MRMESH_API bool dilateRegion( const PointCloud& pointCloud, VertBitSet& region, float dilation, ProgressCallback cb = {}, const AffineXf3f* xf = nullptr );
/// shrinks the region on given euclidian distance. returns false if callback also returns false
MRMESH_API bool erodeRegion( const PointCloud& pointCloud, VertBitSet& region, float erosion, ProgressCallback cb = {}, const AffineXf3f* xf = nullptr );
}