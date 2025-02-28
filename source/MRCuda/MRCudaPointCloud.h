#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

namespace MR::Cuda
{

struct PointCloudDataHolder;

/// copy point cloud-related data to the GPU memory
MRCUDA_API Expected<std::unique_ptr<PointCloudDataHolder>> copyDataFrom( const PointCloud& pc,
    const std::vector<Vector3f>* normals = nullptr );

}
