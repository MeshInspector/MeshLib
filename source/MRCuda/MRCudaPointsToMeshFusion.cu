#include "MRCudaPointsToMeshFusion.cuh"
#include "device_launch_parameters.h"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( const Node3* nodes, const OrderedPoint* orderedPoints, const float3* normals, float* volume, int3 dims, float3 voxelSize, float3 origin, PointsToMeshParameters params )
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

        const float radius = 3 * params.sigma;
        const float radiusSq = radius * radius;

        constexpr int MaxStackSize = 32; // to avoid allocations
        int subtasks[MaxStackSize];
        int stackSize = 0;
        subtasks[stackSize++] = 0;

        auto addSubTask = [&] ( int n )
        {
            float distSq = lengthSq( nodes[n].box.getBoxClosestPointTo( voxelCenter ) - voxelCenter );
            if ( distSq <= radiusSq )
                subtasks[stackSize++] = n;
        };

        addSubTask( 0 );
        const auto inv2SgSq = -0.5f / ( params.sigma * params.sigma );
        while ( stackSize > 0 )
        {
            const auto n = subtasks[--stackSize];
            const auto& node = nodes[n];

            if ( node.leaf() )
            {
                auto [first, last] = node.getLeafPointRange();
                for ( int i = first; i < last; ++i )
                {
                    auto coord = orderedPoints[i].coord;
                    if ( lengthSq( coord - voxelCenter ) <= radiusSq )
                    {
                        const auto distSq =  lengthSq(voxelCenter - coord );
                        const auto w = exp( distSq * inv2SgSq );
                        sumWeight += w;
                        sumDist += dot( normals[i], voxelCenter - coord ) * w;
                    }
                        //foundCallback( orderedPoints[i].id, coord );
                }
                continue;
            }

            addSubTask( node.r ); // look at right node later
            addSubTask( node.l ); // look at left node first
        }

        if ( sumWeight >= params.minWeight )
            volume[id] = sumDist / sumWeight;
    }

    void pointsToDistanceVolumeKernel( const Node3* nodes, const OrderedPoint* points, const float3* normals, SimpleVolume* volume, PointsToMeshParameters params )
    {
        constexpr int maxThreadsPerBlock = 640;
        const int size = volume->dims.x * volume->dims.y * volume->dims.z;

        int numBlocks = (int( size ) + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
        kernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, points, normals, volume->data.data(), volume->dims, volume->voxelSize, volume->origin, params);        
    }
}
}