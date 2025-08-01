#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// represents full point cloud (if region is nullptr) or some portion of point cloud (if region pointer is valid)
struct PointCloudPart
{
    const PointCloud& cloud;
    const VertBitSet* region = nullptr; // nullptr here means all valid points of point cloud

    PointCloudPart( const PointCloud& c, const VertBitSet* bs = nullptr ) noexcept : cloud( c ), region( bs )  {}
};

} // namespace MR
