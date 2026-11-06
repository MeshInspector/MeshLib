#pragma once
#ifndef MR_PARSING_FOR_ANY_BINDINGS

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

namespace MR::Cuda
{

struct PointCloudDataHolder;

/// copy point cloud-related data to the GPU memory
MRCUDA_API Expected<std::unique_ptr<PointCloudDataHolder>> copyDataFrom( const PointCloud& pc, bool copyNormals = false,
    const VertNormals* normals = nullptr );

/// return the amount of GPU memory required for PointCloudDataHolder
MRCUDA_API size_t pointCloudHeapBytes( const PointCloud& pc, bool copyNormals = false,
    const VertNormals* normals = nullptr );

} // namespace MR::Cuda
#endif
