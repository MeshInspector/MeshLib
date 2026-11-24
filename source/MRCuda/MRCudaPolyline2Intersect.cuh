#pragma once

#include "MRCudaInplaceStack.cuh"
#include "MRCudaMath.cuh"
#include "MRCudaPolyline.cuh"

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

__device__ bool isPointInsidePolyline( const float2 point, const Polyline2Data& polyline )
{
    InplaceStack<int, 32> nodesStack;
    nodesStack.push( polyline.root );

    int intersectionCounter = 0;
    while ( !nodesStack.empty() )
    {
        const auto n = nodesStack.top();
        nodesStack.pop();
        if ( Polyline2Data::isLeaf( n ) )
        {
            if ( point.x <= polyline.box( n ).min.x )
            {
                ++intersectionCounter;
                continue;
            }

            const auto ue = polyline.leafId( n );
            const auto& org = polyline.points[polyline.orgs[2 * ue + 0]];
            const auto& dst = polyline.points[polyline.orgs[2 * ue + 1]];

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

        const auto& node = polyline.nodes[n];
        if ( rayBoxIntersect( polyline.box( node.l ), point ) )
            nodesStack.push( node.l );
        if ( rayBoxIntersect( polyline.box( node.r ), point ) )
            nodesStack.push( node.r );
    }

    return ( intersectionCounter % 2 ) == 1;
}

} // namespace MR::Cuda
