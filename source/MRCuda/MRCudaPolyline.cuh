#pragma once

#include "MRCudaMath.cuh"

namespace MR::Cuda
{

/// Data buffers for Polyline2 data: its AABB tree as on CPU, points and the origins of edges
struct Polyline2Data
{
    const Node2* __restrict__ nodes;
    const LineLeaf2* __restrict__ leaves;
    int root; ///< 0, or ~0 if the tree has only one leaf
    const float2* __restrict__ points;
    const int* __restrict__ orgs;

    __device__ static bool isLeaf( int n ) { return n < 0; }
    __device__ Box2 box( int n ) const { return n < 0 ? leaves[~n].box() : nodes[n].box; }
    __device__ int leafId( int n ) const { return leaves[~n].line; }
};

/// Data buffers for Polyline3 data: its AABB tree as on CPU, points and the origins of edges
struct Polyline3Data
{
    const TreeNode3* __restrict__ nodes;
    const LineLeaf3* __restrict__ leaves;
    int root; ///< 0, or ~0 if the tree has only one leaf
    const float3* __restrict__ points;
    const int* __restrict__ orgs;

    __device__ static bool isLeaf( int n ) { return n < 0; }
    __device__ const Box3& box( int n ) const { return n < 0 ? leaves[~n].box : nodes[n].box; }
    __device__ int leafId( int n ) const { return leaves[~n].line; }
};

} // namespace MR::Cuda
