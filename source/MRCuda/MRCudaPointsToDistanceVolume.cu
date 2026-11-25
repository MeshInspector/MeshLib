#include "MRCudaPointsToDistanceVolume.cuh"
#include "MRCudaInplaceStack.cuh"

namespace MR
{
namespace Cuda
{
    __global__ void kernel( PointCloudData pc, float* volume, PointsToDistanceVolumeParams params, size_t chunkSize, size_t chunkOffset )
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

        // every level adds at most 3 tasks
        InplaceStack<int, 49> subtasks;
        auto inBall = [&] ( const Box3 & box )
        {
            return lengthSq( box.getBoxClosestPointTo( voxelCenter ) - voxelCenter ) <= radiusSq;
        };
        if ( inBall( pc.rootBox ) )
            subtasks.push( pc.rootRef );

        const auto inv2SgSq = -params.invSigmaModifier / ( params.sigma * params.sigma );
        while ( !subtasks.empty() )
        {
            const auto ref = subtasks.top();
            subtasks.pop();

            if ( ref < 0 )
            {
                const auto range = pc.leafPointRange( ~ref );
                for ( int i = range.x; i < range.y; ++i )
                {
                    auto coord = pc.points[i].coord;
                    if ( lengthSq( coord - voxelCenter ) <= radiusSq )
                    {
                        const auto distSq =  lengthSq(voxelCenter - coord );
                        const auto w = exp( distSq * inv2SgSq );
                        sumWeight += w;
                        const auto dt = dot( pc.normals[pc.points[i].id], voxelCenter - coord );
                        if ( !params.sqrtAngleWeight )
                            sumDist += dt * w;
                        else
                            sumDist += copysignf( sqrt( abs( dt ) ), dt ) * w;
                    }
                }
                continue;
            }

            // the first child is looked at first
            const auto & node = pc.nodes[ref];
            for ( int i = 3; i >= 0; --i )
                if ( node.child[i] && inBall( node.childBox( i ) ) )
                    subtasks.push( node.child[i] );
        }

        if ( sumWeight >= params.minWeight )
            volume[index] = sumDist / sumWeight;
    }

    void pointsToDistanceVolumeKernel( PointCloudData pc, float* volume, PointsToDistanceVolumeParams params, size_t chunkSize, size_t chunkOffset )
    {
        constexpr int maxThreadsPerBlock = 640;

        auto numBlocks = (unsigned int)( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
        kernel <<< numBlocks, maxThreadsPerBlock >>> ( pc, volume, params, chunkSize, chunkOffset );
    }
} // namespace Cuda
} // namespace MR
