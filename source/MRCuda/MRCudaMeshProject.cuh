#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"

namespace MR
{

namespace Cuda
{

// struct simular to MR::Box3 with minimal needed functions
struct Box3
{
    float3 min;
    float3 max;

    __device__ float3 getBoxClosestPointTo( const float3& pt ) const;
};

// struct simular to AABBTreeNode<FaceTreeTraits3> with minimal needed functions
struct Node3
{
    Box3 box;
    int l, r;

    __device__ bool leaf() const;
    __device__ int leafId() const;
};

// mesh topology data, maby can be simplified to have less data transfers between CPU and GPU
struct HalfEdgeRecord
{
    int next;
    int prev;
    int org;
    int left;
};

// GPU analog of CPU PointOnFace struct
struct PointOnFace
{
    int faceId;
    float3 point;
};

// GPU analog of CPU MeshTriPoint struct
struct MeshTriPoint
{
    int edgeId;
    float a;
    float b;
};

// GPU analog of CPU MeshProjectionResult struct
struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint mtp;
    float distSq;
};

// calls mesh projection kernel for each point in parallel
void meshProjectionKernel( const float3* points,
                           const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace,
                           MeshProjectionResult* resVec, float upDistLimitSq, float loDistLimitSq, size_t size );

}

}