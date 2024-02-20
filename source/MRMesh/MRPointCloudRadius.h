#pragma once
#include "MRMeshFwd.h"

namespace MR
{
struct PointCloud;

/// \brief Finds the radius of ball, so on average that ball contained avgPoints excluding the central point
/// \param samples the number of test points to find given number of samples in each
/// \ingroup PointCloudGroup
MRMESH_API float findAvgPointsRadius( const PointCloud& pointCloud, int avgPoints, int samples = 1024 );

/// expands the region on given euclidian distance. returns false if callback also returns false
MRMESH_API bool dilateRegion( const PointCloud& pointCloud, VertBitSet& region, float dilation, ProgressCallback cb = {}, const AffineXf3f* xf = nullptr );
/// shrinks the region on given euclidian distance. returns false if callback also returns false
MRMESH_API bool erodeRegion( const PointCloud& pointCloud, VertBitSet& region, float erosion, ProgressCallback cb = {}, const AffineXf3f* xf = nullptr );
}