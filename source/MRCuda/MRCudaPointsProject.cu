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
        int ref; // as in SimdNode3::child
        float distSq;
    };
    // every level adds at most 3 tasks
    InplaceStack<SubTask, 49> subtasks;

    auto boxDistSq = [&] ( const Box3 & box )
    {
        return lengthSq( box.getBoxClosestPointTo( pt ) - pt );
    };

    // the node to visit next is not pushed in the stack
    SubTask s{ pc.rootRef, boxDistSq( pc.rootBox ) };
    for (;;)
    {
        if ( s.distSq < result.distSq )
        {
            if ( s.ref < 0 )
            {
                const auto range = pc.leafPointRange( ~s.ref );
                for ( int i = range.x; i < range.y; ++i )
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
            }
            else
            {
                const auto & node = pc.nodes[s.ref];
                // the children closer than the found point ordered by distance, the farthest first
                SubTask ch[4];
                int numCh = 0;
                for ( int i = 0; i < 4 && node.child[i]; ++i )
                {
                    const float d = boxDistSq( node.childBox( i ) );
                    if ( !( d < result.distSq ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].distSq < d; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { node.child[i], d };
                }
                if ( numCh > 0 )
                {
                    // descend in the nearest child, the others wait in the stack
                    for ( int i = 0; i + 1 < numCh; ++i )
                        subtasks.push( ch[i] );
                    s = ch[numCh - 1];
                    continue;
                }
            }
        }
        if ( subtasks.empty() )
            break;
        s = subtasks.top();
        subtasks.pop();
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
