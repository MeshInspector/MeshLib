#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR
{

namespace Cuda
{

// analogue for MR::MeshToDistanceMapParams
struct MeshToDistanceMapParams
{
    float3 xRange;
    float3 yRange;
    float3 direction;
    float3 orgPoint;
    int2 resolution;
    float minValue;
    float maxValue;

    bool useDistanceLimits;
    bool allowNegativeValues;
};

void computeMeshDistanceMapKernel( const MeshTree& tree, const float3* meshPoints, MeshToDistanceMapParams params, IntersectionPrecomputes prec, float shift, float* res, MeshTriPoint* outSamples, size_t chunkSize, size_t chunkOffset );

}

}