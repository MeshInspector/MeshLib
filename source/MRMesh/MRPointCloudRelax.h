#pragma once
#include "MRMeshFwd.h"
#include "MRMeshRelax.h"

namespace MR
{

struct PointCloudRelaxParams : RelaxParams
{
    // radius to find neighbors in,
    // 0.0 - default, 0.1*boundibg box diagonal
    float neighborhoodRadius{ 0.0f };
};

// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
MRMESH_API void relax( PointCloud& pointCloud, const PointCloudRelaxParams& params = {}, SimpleProgressCallback cb = {} );

// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
// do not really keeps volume but tries hard
MRMESH_API void relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params = {}, SimpleProgressCallback cb = {} );

struct PointCloudApproxRelaxParams : PointCloudRelaxParams
{
    RelaxApproxType type{ RelaxApproxType::Planar };
};

// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
// approx neighborhoods
MRMESH_API void relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params = {}, SimpleProgressCallback cb = {} );

}