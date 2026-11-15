#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"
namespace MR
{

namespace Cuda
{

// GPU analog of CPU MeshProjectionResult struct
struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint tp;
    float distSq;
};

// calls mesh projection kernel for each point in parallel
void meshProjectionKernel( const float3* points,
                           const WideNode3* nodes, const float3* meshPoints, const LeafTri3* leafTris,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size );

}

}