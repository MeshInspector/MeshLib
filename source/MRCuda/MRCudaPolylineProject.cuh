#pragma once

#include "MRCudaInplaceStack.cuh"
#include "MRCudaLineSegm.cuh"
#include "MRCudaMath.cuh"

namespace MR::Cuda
{

#pragma warning( push )
#pragma warning( disable: 4324 )
/// CUDA variant of MR::PolylineProjectionResult2
struct PolylineProjectionResult2
{
    int line;
    float2 point;
    float distSq;
};
#pragma warning( pop )

/// CUDA variant of MR::findProjectionOnPolyline2
__device__ inline PolylineProjectionResult2 findProjectionOnPolyline2( const float2 point, const Node2* __restrict__ nodes, const float2* __restrict__ points, const int* __restrict__ orgs, float upDistLimitSq, float loDistLimitSq )
{
    PolylineProjectionResult2 result;
    result.line = -1;
    result.distSq = upDistLimitSq;

    struct SubTask
    {
        int n;
        float distSq;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( SubTask s )
    {
        if ( s.distSq < result.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( int n )
    {
        return SubTask { n, lengthSq( nodes[n].box.getBoxClosestPointTo( point ) - point ) };
    };

    addSubTask( getSubTask( 0 ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();

        if ( result.distSq <= s.distSq )
            continue;

        const auto& node = nodes[s.n];
        if ( node.leaf() )
        {
            const auto ue = node.leafId();
            const auto a = points[orgs[2 * ue + 0]];
            const auto b = points[orgs[2 * ue + 1]];

            const auto proj = closestPointOnLineSegm( point, a, b );
            const auto distSq = lengthSq( proj - point );
            if ( distSq < result.distSq )
            {
                result.line = ue;
                result.point = proj;
                result.distSq = distSq;
                if ( distSq <= loDistLimitSq )
                    break;
            }

            continue;
        }

        const auto sl = getSubTask( node.l );
        const auto sr = getSubTask( node.r );
        if ( sl.distSq < sr.distSq )
        {
            addSubTask( sr );
            addSubTask( sl );
        }
        else
        {
            addSubTask( sl );
            addSubTask( sr );
        }
    }

    return result;
}

/// CUDA variant of MR::findEdgesInBall
template <typename Func>
__device__ void findEdgesInBall( const Node3* __restrict__ nodes, const float3* __restrict__ points, const int* __restrict__ orgs, const float3 center, const float radius, Func callback )
{
    const auto radiusSq = radius * radius;

    struct SubTask
    {
        int n;
        float distSq;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto getSubTask = [&] ( int n )
    {
        return SubTask{ n, lengthSq( nodes[n].box.getBoxClosestPointTo( center ) - center ) };
    };

    subtasks.push( getSubTask( 0 ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();

        const auto& node = nodes[s.n];
        if ( node.leaf() )
        {
            if ( radiusSq < s.distSq )
                continue;

            const auto ue = node.leafId();
            const auto a = points[orgs[2 * ue + 0]];
            const auto b = points[orgs[2 * ue + 1]];

            const auto proj = closestPointOnLineSegm( center, a, b );
            const auto distSq = lengthSq( proj - center );
            if ( distSq <= radiusSq )
                callback( ue, proj, distSq );

            continue;
        }

        subtasks.push( getSubTask( node.r ) ); // look at right node later
        subtasks.push( getSubTask( node.l ) ); // look at left node first
    }
}

} // namespace MR::Cuda
