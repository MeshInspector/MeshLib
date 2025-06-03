#include "MRCudaPointsToDistanceVolume.cuh"
#include "MRCudaInplaceStack.cuh"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( const Node3* nodes, const OrderedPoint* orderedPoints, const float3* normals, float* volume, PointsToDistanceVolumeParams params, size_t chunkSize, size_t chunkOffset )
    {
        const size_t gridSize = size_t( params.dimensions.x ) * params.dimensions.y * params.dimensions.z;
        if ( gridSize == 0 )
        {
            assert( false );
            return;
        }

        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if ( index >= chunkSize )
            return;

        size_t gridIndex = index + chunkOffset;
        if ( gridIndex >= gridSize )
            return;

        const unsigned char quietNan[4] = { 0x00 , 0x00, 0xc0, 0x7f };
        volume[index] = *( float* ) quietNan;

        const size_t sizeXY = size_t( params.dimensions.x ) * params.dimensions.y;
        float3 coord;
        coord.z = int( gridIndex / sizeXY ) + 0.5f;
        int sumZ = int( gridIndex % sizeXY );
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

        InplaceStack<int, 32> subtasks;
        subtasks.push( 0 );

        auto addSubTask = [&] ( int n )
        {
            float distSq = lengthSq( nodes[n].box.getBoxClosestPointTo( voxelCenter ) - voxelCenter );
            if ( distSq <= radiusSq )
                subtasks.push( n );
        };

        addSubTask( 0 );
        const auto inv2SgSq = -0.5f / ( params.sigma * params.sigma );
        while ( !subtasks.empty() )
        {
            const auto n = subtasks.top();
            subtasks.pop();
            const auto& node = nodes[n];

            if ( node.leaf() )
            {
                auto range = node.getLeafPointRange();
                auto first = range.x, last = range.y;
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
            volume[index] = sumDist / sumWeight;
    }

    void pointsToDistanceVolumeKernel( const Node3* nodes, const OrderedPoint* points, const float3* normals, float* volume, PointsToDistanceVolumeParams params, size_t chunkSize, size_t chunkOffset )
    {
        constexpr int maxThreadsPerBlock = 640;

        auto numBlocks = (unsigned int)( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
        kernel <<< numBlocks, maxThreadsPerBlock >>> ( nodes, points, normals, volume, params, chunkSize, chunkOffset );
    }
} // namespace Cuda
} // namespace MR
