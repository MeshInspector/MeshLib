#pragma once

#include "MRCudaInplaceStack.cuh"
#include "MRCudaMath.cuh"

namespace MR::Cuda
{

__device__ inline bool rayBoxIntersect( const Box2 box, const float2 plusXRayStart )
{
    if ( box.max.x <= plusXRayStart.x )
        return false;
    if ( box.max.y <= plusXRayStart.y )
        return false;
    if ( box.min.y > plusXRayStart.y )
        return false;
    return true;
}

__device__ bool isPointInsidePolyline( const float2 point, const Node2* __restrict__ nodes, const float2* __restrict__ points, const int* __restrict__ orgs )
{
    InplaceStack<int, 32> nodesStack;
    nodesStack.push( 0 );

    int intersectionCounter = 0;
    while ( !nodesStack.empty() )
    {
        const auto& node = nodes[nodesStack.top()];
        nodesStack.pop();
        if ( node.leaf() )
        {
            if ( point.x <= node.box.min.x )
            {
                ++intersectionCounter;
                continue;
            }

            const auto ue = node.leafId();
            const auto& org = points[orgs[2 * ue + 0]];
            const auto& dst = points[orgs[2 * ue + 1]];

            const auto yLength = (double)dst.y - (double)org.y;
            if ( yLength != 0. )
            {
                const auto ratio = ( (double)point.y - (double)org.y ) / yLength;
                const auto x = ratio * (double)dst.x + ( 1. - ratio ) * (double)org.x;
                if ( point.x <= (float)x )
                    ++intersectionCounter;
            }

            continue;
        }

        if ( rayBoxIntersect( nodes[node.l].box, point ) )
            nodesStack.push( node.l );
        if ( rayBoxIntersect( nodes[node.r].box, point ) )
            nodesStack.push( node.r );
    }

    return ( intersectionCounter % 2 ) == 1;
}

} // namespace MR::Cuda
