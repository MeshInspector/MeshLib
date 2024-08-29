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
#include "MRLocalTriangulations.h"
#include <cfloat>

namespace MR
{

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud, float radius, const ProgressCallback & progress, OrientNormals orient )
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
        auto n = Vector3f( accum.getBestPlane().n );
        if ( orient != OrientNormals::Smart )
        {
            if ( ( dot( n, pointCloud.points[vid] ) > 0 ) == ( orient == OrientNormals::TowardOrigin ) )
                n = -n;
        }
        normals[vid] = n;
    }, progress ) )
        return {};

    return normals;
}

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud, const AllLocalTriangulations& triangs, const ProgressCallback & progress, OrientNormals orient )
{
    MR_TIMER

    VertNormals normals;
    normals.resizeNoInit( pointCloud.points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        auto n = computeNormal( triangs, pointCloud.points, v );
        if ( orient != OrientNormals::Smart )
        {
            if ( ( dot( n, pointCloud.points[v] ) > 0 ) == ( orient == OrientNormals::TowardOrigin ) )
                n = -n;
        }
        normals[v] = n;
    }, progress ) )
        return {};

    return normals;
}

std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud,
    const Buffer<VertId> & closeVerts, int numNei, const ProgressCallback & progress, OrientNormals orient )
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
        auto n = Vector3f( accum.getBestPlane().n );
        if ( orient != OrientNormals::Smart )
        {
            if ( ( dot( n, pointCloud.points[vid] ) > 0 ) == ( orient == OrientNormals::TowardOrigin ) )
                n = -n;
        }
        normals[vid] = n;
    }, progress ) )
        return {};

    return normals;
}

template<class T>
bool orientNormalsCore( const PointCloud& pointCloud, VertNormals& normals, const T & enumNeis, ProgressCallback progress )
{
    MR_TIMER

    const auto bbox = pointCloud.computeBoundingBox();
    if ( !reportProgress( progress, 0.025f ) )
        return false;

    const auto center = bbox.center();
    const auto maxDistSqToCenter = bbox.size().lengthSq() / 4;

    constexpr auto InvalidWeight = -FLT_MAX;
    using HeapT = Heap<float, VertId>;
    std::vector<HeapT::Element> elements;
    elements.reserve( normals.size() );
    for ( VertId v = 0_v; v < normals.size(); ++v )
        elements.push_back( { v, InvalidWeight } );

    if ( !reportProgress( progress, 0.05f ) )
        return false;

    // fill elements with negative weights: larger weight (smaller by magnitude) for points further from the center
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        const auto dcenter = pointCloud.points[v] - center;
        const auto w = dcenter.lengthSq() - maxDistSqToCenter;
        assert( w <= 0 );
        elements[(int)v].val = w;
        // initially orient points' normals' outside of the center
        if ( dot( normals[v], dcenter ) < 0 )
            normals[v] = -normals[v];
    }, subprogress( progress, 0.05f, 0.075f ) ) )
        return false;

    HeapT heap( std::move( elements ) );

    if ( !reportProgress( progress, 0.1f ) )
        return false;

    progress = subprogress( progress, 0.1f, 1.0f );

    auto enweight = [&]( VertId base, VertId candidate )
    {
        // give positive weight to neighbours, with larger value to close points with close normal directions
        const Vector3f cb = pointCloud.points[base] - pointCloud.points[candidate];
        const auto d = 0.01f * cb.lengthSq() + sqr( dot( cb, normals[base] ) ) + sqr( dot( cb, normals[candidate] ) );
        return d > 0 ? 1 / d : FLT_MAX;
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
            if ( weight > heap.value( v ) )
            {
                heap.setLargerValue( v, weight );
                if ( dot( normals[base], normals[v] ) < 0 )
                    normals[v] = -normals[v];
            }
        } );
    };

    for (;;)
    {
        auto [v, weight] = heap.top();
        if ( weight == InvalidWeight )
            break;
        heap.setSmallerValue( v, InvalidWeight );
        enqueueNeighbors( v );
        if ( !reportProgress( progress, [&] { return (float)visitedCount / totalCount; }, visitedCount, 0x10000 ) )
            return false;
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

bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, const AllLocalTriangulations& triangs,
     const ProgressCallback & progress )
{
    MR_TIMER
    return orientNormalsCore( pointCloud, normals,
        [&triangs]( VertId v, auto callback )
        {
            const auto * p = triangs.neighbors.data() + triangs.fanRecords[v].firstNei;
            const auto * pEnd = triangs.neighbors.data() + triangs.fanRecords[v+1].firstNei;
            for ( ; p < pEnd; ++p )
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

std::optional<VertNormals> makeOrientedNormals( const PointCloud& pointCloud,
    AllLocalTriangulations& triangs, const ProgressCallback & progress )
{
    MR_TIMER

    if ( !autoOrientLocalTriangulations( pointCloud, triangs, pointCloud.validPoints, subprogress( progress, 0.0f, 0.9f ) ) )
        return {};

    // since triangulations are oriented then normals will be oriented as well
    return makeUnorientedNormals( pointCloud, triangs, subprogress( progress, 0.9f, 1.0f ) );
}

VertNormals makeNormals( const PointCloud& pointCloud, int avgNeighborhoodSize )
{
    return *makeOrientedNormals( pointCloud, findAvgPointsRadius( pointCloud, avgNeighborhoodSize ) );
}

} //namespace MR
