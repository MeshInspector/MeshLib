#pragma once

#include "exports.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPointsProject.h"

namespace MR::Cuda
{

/// computes the closest points on point cloud to given points
MRCUDA_API Expected<std::vector<MR::PointsProjectionResult>> findProjectionOnPoints( const PointCloud& pointCloud,
    const std::vector<Vector3f>& points, const AffineXf3f* pointsXf, const AffineXf3f* refXf, float upDistLimitSq,
    float loDistLimitSq );

/// returns the minimal amount of free GPU memory required for \ref MR::Cuda::findProjectionOnPoints
MRMESH_API size_t findProjectionOnPointsHeapBytes( const PointCloud& pointCloud, size_t pointsCount );

} // namespace MR::Cuda
