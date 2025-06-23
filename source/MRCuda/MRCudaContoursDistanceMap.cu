#include "MRCudaContoursDistanceMap.cuh"
#include "MRCudaBasic.h"
#include "MRCudaFloat.cuh"
#include "MRCudaInplaceStack.cuh"
#include "MRCudaLineSegm.cuh"

#include <float.h>

namespace MR
{
namespace Cuda
{

__global__ void kernel(
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Polyline2Data polyline,
    float* dists, const size_t chunkSize, size_t chunkOffset )
{
    if ( chunkSize == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    float2 pt;

    size_t gridIndex = index + chunkOffset;
    size_t x = gridIndex % resolution.x;
    size_t y = gridIndex / resolution.x;

    pt.x = pixelSize.x * x + originPoint.x;
    pt.y = pixelSize.y * y + originPoint.y;

    float resDistSq = FLT_MAX;

    struct SubTask
    {
        int n;
        float distSq;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < resDistSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( int n )
    {
        return SubTask{ n, lengthSq( polyline.nodes[n].box.getBoxClosestPointTo( pt ) - pt ) };
    };

    addSubTask( getSubTask( 0 ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = polyline.nodes[s.n];
        if ( s.distSq >= resDistSq )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            float2 a = polyline.points[polyline.orgs[2 * lineId]];
            float2 b = polyline.points[polyline.orgs[2 * lineId + 1]];
            auto proj = closestPointOnLineSegm( pt, a, b );

            float distSq = lengthSq( proj - pt );
            if ( distSq < resDistSq )
            {
                resDistSq = distSq;
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

    dists[index] = sqrt( resDistSq );
}

void contoursDistanceMapProjectionKernel( 
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Polyline2Data polyline, float* dists,
    const size_t chunkSize, size_t chunkOffset )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );

    // kernel
    kernel<<< numBlocks, maxThreadsPerBlock >>>(
        originPoint, resolution, pixelSize,
        polyline, dists, chunkSize, chunkOffset );
}

}
}