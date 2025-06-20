#include "MRCudaContoursDistanceMap.cuh"
#include "MRCudaBasic.h"
#include "MRCudaFloat.cuh"
#include "MRCudaInplaceStack.cuh"

#include <float.h>

namespace MR
{
namespace Cuda
{

__device__ float2 closestPointOnLineSegm( const float2& pt, const float2& a, const float2& b )
{
    const auto ab = b - a;
    const auto dt = dot( pt - a, ab );
    const auto abLengthSq = lengthSq( ab );
    if ( dt <= 0 )
        return a;
    if ( dt >= abLengthSq )
        return b;
    auto ratio = dt / abLengthSq;
    return a * ( 1 - ratio ) + b * ratio;
}

__global__ void kernel(
    const float2 originPoint, const int2 resolution, const float2 pixelSize,
    const Node2* __restrict__ nodes, const float2* __restrict__ polylinePoints, const int* __restrict__ orgs,
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
        return SubTask{ n, lengthSq( nodes[n].box.getBoxClosestPointTo( pt ) - pt ) };
    };

    addSubTask( getSubTask( 0 ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = nodes[s.n];
        if ( s.distSq >= resDistSq )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            float2 a = polylinePoints[orgs[2 * lineId]];
            float2 b = polylinePoints[orgs[2 * lineId + 1]];
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
    const Node2* nodes, const float2* polylinePoints, const int* orgs, float* dists,
    const size_t chunkSize, size_t chunkOffset )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );

    // kernel
    kernel<<< numBlocks, maxThreadsPerBlock >>>(
        originPoint, resolution, pixelSize,
        nodes, polylinePoints, orgs, dists, chunkSize, chunkOffset );
}

}
}