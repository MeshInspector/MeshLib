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

} // namespace MR::Cuda
