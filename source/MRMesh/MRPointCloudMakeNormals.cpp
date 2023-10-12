#include "MRPointCloudMakeNormals.h"
#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRId.h"
#include "MRBox.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRPointsInBall.h"
#include "MRTimer.h"
#include "MRPlane3.h"
#include "MRPointCloudRadius.h"
#include <cfloat>
#include <queue>

namespace MR
{

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud, float radius, const ProgressCallback & progress )
{
    MR_TIMER

    VertNormals normals( pointCloud.points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId vid )
    {
        PointAccumulator accum;
        findPointsInBall( pointCloud, pointCloud.points[vid], radius, [&]( VertId, const Vector3f& coord )
        {
            accum.addPoint( Vector3d( coord ) );
        } );
        normals[vid] = Vector3f( accum.getBestPlane().n ).normalized();
    }, progress ) )
        return {};

    return normals;
}

struct NormalCandidate
{
    NormalCandidate() = default;
    NormalCandidate(VertId _id, VertId _baseId,float w ):
        id{_id}, baseId{_baseId}, weight{w}{}
    VertId id;
    VertId baseId;
    float weight{FLT_MAX};
};

inline bool operator < ( const NormalCandidate& l, const NormalCandidate& r )
{
    return l.weight > r.weight;
}

bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, float radius, const ProgressCallback & progress )
{
    MR_TIMER

    VertScalars minWeights( normals.size(), FLT_MAX );
    std::priority_queue<NormalCandidate> queue;

    auto enweight = [&]( VertId base, VertId candidate )
    {
        Vector3f cb = pointCloud.points[base] - pointCloud.points[candidate];
        return 0.01f * cb.lengthSq() + sqr( dot( cb, normals[base] ) ) + sqr( dot( cb, normals[candidate] ) );
    };

    VertBitSet notVisited = pointCloud.validPoints;
    const auto totalCount = notVisited.count();
    size_t visitedCount = 0;

    auto enqueueNeighbors = [&]( VertId base )
    {
        assert( notVisited.test( base ) );
        notVisited.reset( base );
        ++visitedCount;
        findPointsInBall( pointCloud, pointCloud.points[base], radius,
                          [&]( VertId v, const Vector3f& )
        {
            if ( v == base )
                return;
            if ( !notVisited.test( v ) )
                return;
            float weight = enweight( base, v );
            if ( weight < minWeights[v] )
            {
                queue.emplace( v, base, weight );
                minWeights[v] = weight;
            }
        } );
    };

    auto findFirst = [&]()->VertId
    {
        MR_TIMER;
        VertId xMostVert = {};
        float maxX = -FLT_MAX;
        for ( auto v : notVisited )
        {
            assert( minWeights[v] == FLT_MAX );
            auto xDot = dot( pointCloud.points[v], Vector3f::plusX() );
            if ( xDot > maxX )
            {
                xMostVert = v;
                maxX = xDot;
            }
        }
        if ( xMostVert )
        {
            minWeights[xMostVert] = 0.0f;
            if ( dot( normals[xMostVert], Vector3f::plusX() ) < 0.0f )
                normals[xMostVert] = -normals[xMostVert];
        }
        return xMostVert;
    };

    NormalCandidate current;
    VertId first = findFirst();
    while ( first.valid() )
    {
        enqueueNeighbors( first );
        while ( !queue.empty() )
        {
            current = queue.top(); // cannot use std::move unfortunately since top() returns const reference
            queue.pop();
            if ( current.weight > minWeights[current.id] )
                continue;
            if ( dot( normals[current.baseId], normals[current.id] ) < 0.0f )
                normals[current.id] = -normals[current.id];
            enqueueNeighbors( current.id );
            if ( !reportProgress( progress, [&] { return (float)visitedCount / totalCount; }, visitedCount, 1024 ) )
                return false;
        }
        first = findFirst();
    }
    return true;
}

std::optional<VertNormals> makeOrientedNormals( const PointCloud& pointCloud,
    float radius, const ProgressCallback & progress )
{
    MR_TIMER

    auto optNormals = makeUnorientedNormals( pointCloud, radius, subprogress( progress, 0.0f, 0.1f ) );
    if ( !optNormals )
        return optNormals;

    if ( !orientNormals( pointCloud, *optNormals, radius, subprogress( progress, 0.1f, 1.0f ) ) )
        optNormals.reset();

    return optNormals;
}

VertNormals makeNormals( const PointCloud& pointCloud, int avgNeighborhoodSize )
{
    return *makeOrientedNormals( pointCloud, findAvgPointsRadius( pointCloud, avgNeighborhoodSize ) );
}

} //namespace MR
