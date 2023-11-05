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
#include "MRPointsProject.h"
#include "MRHeap.h"
#include "MRBuffer.h"
#include <cfloat>

namespace MR
{

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud, float radius, const ProgressCallback & progress )
{
    MR_TIMER

    VertNormals normals;
    normals.resizeNoInit( pointCloud.points.size() );
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

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud,
    const Buffer<VertId> & closeVerts, int numNei, const ProgressCallback & progress )
{
    MR_TIMER

    VertNormals normals;
    normals.resizeNoInit( pointCloud.points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId vid )
    {
        PointAccumulator accum;
        accum.addPoint( pointCloud.points[vid] );
        VertId * p = closeVerts.data() + ( (size_t)vid * numNei );
        const VertId * pEnd = p + numNei;
        for ( ; p < pEnd && *p; ++p )
            accum.addPoint( pointCloud.points[*p] );
        normals[vid] = Vector3f( accum.getBestPlane().n ).normalized();
    }, progress ) )
        return {};

    return normals;
}

struct NormalCandidate
{
    VertId baseId;
    float weight = FLT_MAX;
};

inline bool operator < ( const NormalCandidate& l, const NormalCandidate& r )
{
    return l.weight > r.weight;
}

template<class T>
bool orientNormalsCore( const PointCloud& pointCloud, VertNormals& normals, const T & enumNeis, const ProgressCallback & progress )
{
    MR_TIMER

    Heap<NormalCandidate, VertId> heap( normals.size() );

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
        enumNeis( base, [&]( VertId v )
        {
            assert ( v != base );
            if ( !notVisited.test( v ) )
                return;
            float weight = enweight( base, v );
            if ( weight < heap.value( v ).weight )
                heap.setLargerValue( v, NormalCandidate{ base, weight } );
        } );
    };

    auto findFirst = [&]()->VertId
    {
        MR_TIMER
        // find not visited point with the largest x-coordinate
        VertId xMostVert;
        float maxX = -FLT_MAX;
        for ( auto v : notVisited )
        {
            assert( heap.value( v ).weight == FLT_MAX );
            auto xDot = dot( pointCloud.points[v], Vector3f::plusX() );
            if ( xDot > maxX )
            {
                xMostVert = v;
                maxX = xDot;
            }
        }
        if ( xMostVert )
        {
            // orient the point with the largest x-coordinate to have normal outside
            if ( dot( normals[xMostVert], Vector3f::plusX() ) < 0.0f )
                normals[xMostVert] = -normals[xMostVert];
        }
        return xMostVert;
    };

    VertId first = findFirst();
    while ( first.valid() )
    {
        enqueueNeighbors( first );
        for (;;)
        {
            auto [v, c] = heap.top();
            if ( c.weight == FLT_MAX )
                break;
            heap.setSmallerValue( v, NormalCandidate{ {}, FLT_MAX } );
            if ( dot( normals[c.baseId], normals[v] ) < 0.0f )
                normals[v] = -normals[v];
            enqueueNeighbors( v );
            if ( !reportProgress( progress, [&] { return (float)visitedCount / totalCount; }, visitedCount, 0x10000 ) )
                return false;
        }
        first = findFirst();
    }
    return true;
}

bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, float radius, const ProgressCallback & progress )
{
    return orientNormalsCore( pointCloud, normals,
        [&]( VertId base, auto callback )
        {
            findPointsInBall( pointCloud, pointCloud.points[base], radius,
                [&]( VertId v, const Vector3f& )
                {
                    if ( v == base )
                        return;
                    callback( v );
                } );
        }, progress );
}

bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, const Buffer<VertId> & closeVerts, int numNei,
    const ProgressCallback & progress )
{
    return orientNormalsCore( pointCloud, normals,
        [&]( VertId base, auto callback )
        {
            VertId * p = closeVerts.data() + ( (size_t)base * numNei );
            const VertId * pEnd = p + numNei;
            for ( ; p < pEnd && *p; ++p )
                callback( *p );
        }, progress );
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
