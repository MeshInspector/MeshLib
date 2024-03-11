#include "MRCudaPointsToMeshFusion.cuh"
#include "device_launch_parameters.h"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( const Node3* nodes, const float3* points, const float3* normals, float* volume, int3 dims, float3 voxelSize, float3 origin, PointsToMeshParameters params )
    {
        const int size = dims.x * dims.y * dims.z;
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= size )
            return;

        const int sizeXY = dims.x * dims.y;
        float3 coord;
        coord.z = int( id / sizeXY ) + 0.5f;
        int sumZ = int( id % sizeXY );
        coord.y = sumZ / dims.x + 0.5f;
        coord.x = sumZ % dims.x + 0.5f;

        float3 voxelCenter = origin;
        voxelCenter.x += voxelSize.x * coord.x;
        voxelCenter.y += voxelSize.y * coord.y;
        voxelCenter.z += voxelSize.z * coord.z;

        float sumDist = 0;
        float sumWeight = 0;
    }

    void pointsToDistanceVolumeKernel( const Node3* nodes, const float3* points, const float3* normals, SimpleVolume* volume, PointsToMeshParameters params )
    {
        constexpr int maxThreadsPerBlock = 640;
        const int size = volume->dims.x * volume->dims.y * volume->dims.z;

        int numBlocks = (int( size ) + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
        kernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, points, normals, volume->data.data(), volume->dims, volume->voxelSize, volume->origin, params);
    }
}
}