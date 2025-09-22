#pragma once

#include "config.h"
#ifndef MRCUDA_NO_VOXELS
#include "exports.h"

#include "MRVoxels/MRSweptVolume.h"

#include <span>

namespace MR::Cuda
{

/// CUDA implementation of tool distance computation
class MRCUDA_CLASS ComputeToolDistance : public IComputeToolDistance
{
public:
    MRCUDA_API ComputeToolDistance();
    MRCUDA_API ~ComputeToolDistance() override;

    MRCUDA_API Expected<Vector3i> prepare( const Vector3i& dims, const Polyline3& toolpath,
        const EndMillTool& toolSpec ) override;
    MRCUDA_API Expected<Vector3i> prepare( const Vector3i& dims, const Polyline3& toolpath,
        const Polyline2& toolProfile ) override;

    MRCUDA_API Expected<void> computeToolDistance( std::span<float> output, const Vector3i& dims,
        float voxelSize, const Vector3f& origin, float padding ) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace MR::Cuda

#endif
