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

void findProjectionOnPointsKernel( PointsProjectionResult* __restrict__ res, PointCloudData pc,
    const float3* __restrict__ points, const uint64_t* __restrict__ validPoints, Matrix4 xf, float upDistLimitSq,
    float loDistLimitSq, bool skipSameIndex, size_t chunkSize, size_t chunkOffset );

} // namespace MR::Cuda
