#include "MRCudaPointsProject.cuh"
#include "MRCudaInplaceStack.cuh"

namespace MR::Cuda
{

__global__ void kernel( PointsProjectionResult* __restrict__ res, PointCloudData pc, const float3* __restrict__ points,
    const uint64_t* __restrict__ validPoints, Matrix4 xf, float upDistLimitSq, float loDistLimitSq, bool skipSameIndex,
    size_t chunkSize, size_t chunkOffset )
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    const auto globalIndex = index + chunkOffset;
    if ( validPoints && !testBit( validPoints, globalIndex ) )
        return;

    const auto pt = xf.isIdentity ? points[index] : xf.transform( points[index] );

    PointsProjectionResult result;
    result.distSq = upDistLimitSq;
    result.vertId = -1;

    struct SubTask
    {
        int n;
        float distSq;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < result.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( int n )
    {
        const auto box = pc.nodes[n].box;
        const auto distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( 0 ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = pc.nodes[s.n];
        if ( s.distSq >= result.distSq )
            continue;

        if ( node.leaf() )
        {
            auto range = node.getLeafPointRange();
            auto begin = range.x, end = range.y;
            for ( int i = begin; i < end; ++i )
            {
                if ( skipSameIndex && i == globalIndex )
                    continue;

                const auto proj = pc.points[i].coord;
                const auto distSq = lengthSq( proj - pt );
                if ( distSq < result.distSq )
                {
                    result = { distSq, pc.points[i].id };
                    if ( distSq <= loDistLimitSq )
                        goto exit;
                }
            }
            continue;
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
    const uint64_t* validPoints, Matrix4 xf, float upDistLimitSq, float loDistLimitSq, bool skipSameIndex,
    size_t chunkSize, size_t chunkOffset )
{
    constexpr int maxThreadsPerBlock = 640;
    const auto numBlocks = (int)( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel <<< numBlocks, maxThreadsPerBlock >>> ( res, pc, points, validPoints, xf, upDistLimitSq, loDistLimitSq, skipSameIndex, chunkSize, chunkOffset );
}

} // namespace MR::Cuda
