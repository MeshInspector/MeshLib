#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRRelaxParams.h"

namespace MR
{

/// \addtogroup PointCloudGroup
/// \{

struct PointCloudRelaxParams : RelaxParams
{
    /// radius to find neighbors in,
    /// 0.0 - default, 0.1*boundibg box diagonal
    float neighborhoodRadius{ 0.0f };
};

/// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( PointCloud& pointCloud, const PointCloudRelaxParams& params = {}, ProgressCallback cb = {} );

/// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
/// do not really keeps volume but tries hard
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params = {}, ProgressCallback cb = {} );

struct PointCloudApproxRelaxParams : PointCloudRelaxParams
{
    RelaxApproxType type{ RelaxApproxType::Planar };
};

/// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
/// approx neighborhoods
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params = {}, ProgressCallback cb = {} );

/// \}

}
