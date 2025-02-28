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

// point cloud data required for algorithms
struct PointCloudData
{
    const Node3* __restrict__ nodes;
    const OrderedPoint* __restrict__ points;
    const float3* __restrict__ normals;
};

// GPU memory holder for point cloud data
struct PointCloudDataHolder
{
    DynamicArray<Node3> nodes;
    DynamicArray<OrderedPoint> points;
    DynamicArray<float3> normals;

    [[nodiscard]] PointCloudData data() const
    {
        return {
            nodes.data(),
            points.data(),
            normals.data(),
        };
    }
};

} // namespace MR::Cuda
