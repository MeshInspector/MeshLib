#pragma once

#include "MRCudaMath.cuh"

namespace MR::Cuda
{

/// Data buffers for Polyline2 data
struct Polyline2Data
{
    const Node2* __restrict__ nodes;
    const float2* __restrict__ points;
    const int* __restrict__ orgs;
};

/// Data buffers for Polyline3 data
struct Polyline3Data
{
    const Node3* __restrict__ nodes;
    const float3* __restrict__ points;
    const int* __restrict__ orgs;
};

} // namespace MR::Cuda
