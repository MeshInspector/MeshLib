#pragma once

#include "MRCudaMath.cuh"
#include "MRCudaPolyline.cuh"

namespace MR
{

namespace Cuda
{

// call polyline projection kernel for each distance map pixel in parallel
void contoursDistanceMapProjectionKernel(
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Polyline2Data polyline, float* dists,
    const size_t chunkSize, size_t chunkOffset );

}

}