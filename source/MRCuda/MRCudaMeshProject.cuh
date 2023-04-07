#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.h"

namespace MR
{

namespace Cuda
{
struct Box3
{
    float3 min;
    float3 max;

    __device__ float3 getBoxClosestPointTo( const float3& pt ) const;
};

struct Node3
{
    Box3 box;
    int l, r;

    __device__ bool leaf() const;
    __device__ int leafId() const;
};

struct HalfEdgeRecord
{
    int next;
    int prev;
    int org;
    int left;
};

struct PointOnFace
{
    int faceId;
    float3 point;
};

struct MeshTriPoint
{
    int edgeId;
    float a;
    float b;
};

struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint mtp;
    float distSq;
};

void meshProjectionKernel( const float3* points,
                           const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace,
                           MeshProjectionResult* resVec, float upDistLimitSq, float loDistLimitSq, size_t size );

}

}