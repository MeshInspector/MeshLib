#pragma once

#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR::Cuda
{

// struct similar to MR::Point
struct OrderedPoint
{
    float3 coord;
    int id;
};

/// the number of points in a leaf of AABBTreePoints: all leaves are full except the last one
constexpr int cMaxLeafPoints = 16;

// point cloud data required for algorithms: its AABBTreePoints in its own layout
struct PointCloudData
{
    const SimdNode3* __restrict__ nodes; ///< the internal nodes, the root is the first one (if the tree has more than one leaf)
    const OrderedPoint* __restrict__ points; ///< the points of leaf l are [cMaxLeafPoints * l, cMaxLeafPoints * ( l + 1 ) )
    const float3* __restrict__ normals;
    Box3 rootBox; ///< the box of all points
    int rootRef; ///< the reference to the root as in SimdNode3::child: 0, or ~0 if the tree has one leaf only
    int numPoints;

    /// returns [first,last) indices of the points of given leaf
    __device__ int2 leafPointRange( int leaf ) const
    {
        const int first = leaf * cMaxLeafPoints;
        const int last = first + cMaxLeafPoints;
        return { first, last < numPoints ? last : numPoints };
    }
};

// GPU memory holder for point cloud data
struct PointCloudDataHolder
{
    DynamicArray<SimdNode3> nodes;
    DynamicArray<OrderedPoint> points;
    DynamicArray<float3> normals;
    Box3 rootBox{};
    int rootRef = 0;
    int numPoints = 0;

    [[nodiscard]] PointCloudData data() const
    {
        return {
            nodes.data(),
            points.data(),
            normals.data(),
            rootBox,
            rootRef,
            numPoints,
        };
    }
};

} // namespace MR::Cuda
