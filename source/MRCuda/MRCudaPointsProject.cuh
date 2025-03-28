#pragma once

#include "MRCudaPointCloud.cuh"

namespace MR::Cuda
{

// GPU analog of CPU PointsProjectionResult struct
struct PointsProjectionResult
{
    float distSq;
    int vertId;
};

void findProjectionOnPointsKernel( PointsProjectionResult* res, PointCloudData pc, const float3* points,
    const uint64_t* validPoints, Matrix4 xf, float upDistLimitSq, float loDistLimitSq, bool skipSameIndex,
    size_t chunkSize, size_t chunkOffset );

} // namespace MR::Cuda
