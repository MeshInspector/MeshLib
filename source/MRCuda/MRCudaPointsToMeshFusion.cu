#include "MRCudaPointsToMeshFusion.cuh"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( const Node3* nodes, const float3* points, const float3* normals, float* volume, int3 dims, float3 voxelSize, PointsToMeshParameters params )
    {

    }

    void pointsToDistanceVolumeKernel( const Node3* nodes, const float3* points, const float3* normals, SimpleVolume* volume, PointsToMeshParameters params )
    {
        constexpr int maxThreadsPerBlock = 640;
        const int size = volume->dims.x * volume->dims.y * volume->dims.z;

        int numBlocks = (int( size ) + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
        kernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, points, normals, volume->data.data(), volume->dims, volume->voxelSize, params);
    }
}
}