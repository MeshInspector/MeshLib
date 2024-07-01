#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"

namespace MR
{

namespace Cuda
{

// struct simular to MR::Box2 with minimal needed functions
struct Box2
{
    float2 min;
    float2 max;

    __device__ float2 getBoxClosestPointTo( const float2& pt ) const;
};

// struct simular to AABBTreeNode<LineTreeTraits<Vector2f>> with minimal needed functions
struct Node2
{
    Box2 box;
    int l, r;

    __device__ bool leaf() const;
    __device__ int leafId() const;
};

// call polyline projection kerenel for each distance map pixel in parallel
void contoursDistanceMapProjectionKernel(
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Node2* nodes, const float2* polylinePoints, const int* orgs, float* dists,
    const size_t size );

}

}