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

cudaError_t computeMeshDistanceMapKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, MeshToDistanceMapParams params, IntersectionPrecomputes prec, float shift, float* res, MeshTriPoint* outSamples );

}

}