#pragma once

#include "MRCudaMath.cuh"

namespace MR
{

namespace Cuda
{

// call polyline projection kerenel for each distance map pixel in parallel
void contoursDistanceMapProjectionKernel(
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Node2* nodes, const float2* polylinePoints, const int* orgs, float* dists,
    const size_t chunkSize, size_t chunkOffset );

}

}