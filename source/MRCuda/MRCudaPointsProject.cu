#include "MRCudaPointsProject.cuh"

#include "device_launch_parameters.h"

namespace MR::Cuda
{

__global__ void kernel( PointsProjectionResult* __restrict__ res, PointCloudData pc, const float3* __restrict__ points,
    Matrix4 pointsXf, Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t chunkSize, size_t chunkOffset )
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    const auto pt = pointsXf.isIdentity ? points[index] : pointsXf.transform( points[index] );

    PointsProjectionResult result {
        .distSq = upDistLimitSq,
        .vertId = -1,
    };

    struct SubTask
    {
        int n;
        float distSq;
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < result.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( int n )
    {
        const auto box = pc.nodes[n].box;
        const auto distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( 0 ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = pc.nodes[s.n];
        if ( s.distSq >= result.distSq )
            continue;

        if ( node.leaf() )
        {
            auto [begin, end] = node.getLeafPointRange();
            for ( int i = begin; i < end; ++i )
            {
                const auto proj = refXf.isIdentity ? pc.points[i].coord : refXf.transform( pc.points[i].coord );
                const auto distSq = lengthSq( proj - pt );
                if ( distSq < result.distSq )
                {
                    result = { distSq, pc.points[i].id };
                    if ( distSq <= loDistLimitSq )
                        goto exit;
                }
            }
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.distSq < s2.distSq )
        {
            const auto temp = s1;
            s1 = s2;
            s2 = temp;
        }
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }

exit:
    res[index] = result;
}

void findProjectionOnPointsKernel( PointsProjectionResult* res, PointCloudData pc, const float3* points,
    Matrix4 pointsXf, Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t chunkSize, size_t chunkOffset )
{
    constexpr int maxThreadsPerBlock = 640;
    const auto numBlocks = (int)( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel <<< numBlocks, maxThreadsPerBlock >>> ( res, pc, points, pointsXf, refXf, upDistLimitSq, loDistLimitSq, chunkSize, chunkOffset );
}

} // namespace MR::Cuda
