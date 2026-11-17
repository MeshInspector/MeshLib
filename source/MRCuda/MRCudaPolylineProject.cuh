#pragma once

#include "MRCudaInplaceStack.cuh"
#include "MRCudaLineSegm.cuh"
#include "MRCudaMath.cuh"
#include "MRCudaPolyline.cuh"

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
__device__ inline PolylineProjectionResult2 findProjectionOnPolyline2( const float2 point, const Polyline2Data& polyline, float upDistLimitSq, float loDistLimitSq )
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
        return SubTask { n, lengthSq( polyline.box( n ).getBoxClosestPointTo( point ) - point ) };
    };

    addSubTask( getSubTask( polyline.root ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();

        if ( result.distSq <= s.distSq )
            continue;

        if ( Polyline2Data::isLeaf( s.n ) )
        {
            const auto ue = polyline.leafId( s.n );
            const auto a = polyline.points[polyline.orgs[2 * ue + 0]];
            const auto b = polyline.points[polyline.orgs[2 * ue + 1]];

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

        const auto& node = polyline.nodes[s.n];
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
__device__ void findEdgesInBall( const Polyline3Data& polyline, const float3 center, const float radius, Func callback )
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
        return SubTask{ n, lengthSq( polyline.box( n ).getBoxClosestPointTo( center ) - center ) };
    };

    subtasks.push( getSubTask( polyline.root ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();

        if ( Polyline3Data::isLeaf( s.n ) )
        {
            if ( radiusSq < s.distSq )
                continue;

            const auto ue = polyline.leafId( s.n );
            const auto a = polyline.points[polyline.orgs[2 * ue + 0]];
            const auto b = polyline.points[polyline.orgs[2 * ue + 1]];

            const auto proj = closestPointOnLineSegm( center, a, b );
            const auto distSq = lengthSq( proj - center );
            if ( distSq <= radiusSq )
                callback( ue, proj, distSq );

            continue;
        }

        const auto& node = polyline.nodes[s.n];
        subtasks.push( getSubTask( node.r ) ); // look at right node later
        subtasks.push( getSubTask( node.l ) ); // look at left node first
    }
}

} // namespace MR::Cuda
