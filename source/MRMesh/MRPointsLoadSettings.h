#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// structure with settings and side output parameters for loading point cloud
struct PointsLoadSettings
{
    VertColors* colors = nullptr; ///< points where to load point color map
    AffineXf3f* outXf = nullptr; ///< transform for the loaded point cloud
    ProgressCallback callback; ///< callback for set progress and stop process
};

} // namespace MR
