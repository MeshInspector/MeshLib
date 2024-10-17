#include "MRCudaPointsToDistanceVolume.cuh"
#include "device_launch_parameters.h"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( const Node3* nodes, const OrderedPoint* orderedPoints, const float3* normals, float* volume, PointsToDistanceVolumeParams params )
    {
        const size_t size = size_t( params.dimensions.x ) * params.dimensions.y * params.dimensions.z;
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= size )
            return;

        const unsigned char quietNan[4] = { 0x00 , 0x00, 0xc0, 0x7f };
        volume[id] = *( float* ) quietNan;

        const size_t sizeXY = size_t( params.dimensions.x ) * params.dimensions.y;
        float3 coord;
        coord.z = int( id / sizeXY ) + 0.5f;
        int sumZ = int( id % sizeXY );
        coord.y = sumZ / params.dimensions.x + 0.5f;
        coord.x = sumZ % params.dimensions.x + 0.5f;

        float3 voxelCenter = params.origin;
        voxelCenter.x += params.voxelSize.x * coord.x;
        voxelCenter.y += params.voxelSize.y * coord.y;
        voxelCenter.z += params.voxelSize.z * coord.z;

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
                        sumDist += dot( normals[orderedPoints[i].id], voxelCenter - coord ) * w;
                    }
                }
                continue;
            }

            addSubTask( node.r ); // look at right node later
            addSubTask( node.l ); // look at left node first
        }

        if ( sumWeight >= params.minWeight )
            volume[id] = sumDist / sumWeight;
    }

    bool pointsToDistanceVolumeKernel( const Node3* nodes, const OrderedPoint* points, const float3* normals, float* volume, PointsToDistanceVolumeParams params )
    {
        constexpr int maxThreadsPerBlock = 640;
        const size_t size = size_t( params.dimensions.x ) * params.dimensions.y * params.dimensions.z;

        auto numBlocks = (unsigned int)( ( size_t( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
        kernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, points, normals, volume, params );
        return ( cudaGetLastError() == cudaSuccess );
    }
} // namespace Cuda
} // namespace MR
