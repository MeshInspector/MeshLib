#include "MRPointCloudRadius.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsInBall.h"
#include "MRParallelFor.h"
#include "MRPointsProject.h"
#include "MRFewSmallest.h"
#include "MRTimer.h"
#include <numeric>

namespace MR
{

float findAvgPointsRadius( const PointCloud& pointCloud, int avgPoints, int samples )
{
    MR_TIMER

    assert( avgPoints > 0 );
    assert( samples > 0 );
    const auto totalPoints = (int)pointCloud.validPoints.count();
    std::vector<VertId> sampleIds;
    sampleIds.reserve( samples );
    int s = totalPoints;
    for ( auto v : pointCloud.validPoints )
    {
        if ( s >= totalPoints )
        {
            sampleIds.push_back( v );
            s -= totalPoints;
        }
        s += samples;
    }
    if ( sampleIds.empty() )
        return 0;

    tbb::enumerable_thread_specific<FewSmallest<PointsProjectionResult>> perThreadNeis( avgPoints + 1 );

    pointCloud.getAABBTree(); // to avoid multiple calls to tree construction from parallel region,
                      // which can result that two different vertices will start being processed by one thread

    std::vector<float> radia( sampleIds.size() );

    ParallelFor( sampleIds, [&]( size_t i )
    {
        const VertId v = sampleIds[i];
        auto & neis = perThreadNeis.local();
        neis.clear();
        assert( neis.maxElms() == avgPoints + 1 );
        findFewClosestPoints( pointCloud.points[v], pointCloud, neis );
        radia[i] = neis.empty() ? 0.0f : std::sqrt( neis.top().distSq );
    } );

    return std::accumulate( radia.begin(), radia.end(), 0.0f ) / radia.size();
}

bool dilateRegion( const PointCloud& pointCloud, VertBitSet& region, float dilation, ProgressCallback cb, const AffineXf3f* xf )
{
    auto regionCopy = region;

    const auto res =  BitSetParallelForAll( region, [&] ( VertId testVertex )
    {
        if ( regionCopy.test( testVertex ) )
            return;

        const Vector3f point = xf ? (*xf)( pointCloud.points[testVertex] ) : pointCloud.points[testVertex];

        findPointsInBall( pointCloud, point, dilation, [&] ( VertId v, const Vector3f& )
        {
            if ( regionCopy.test( testVertex ) )
                return;

            if ( region.test( v ) )
                regionCopy.set( testVertex, true );
        }, xf );
    }, cb );

    if ( !res )
        return false;

    region = regionCopy;
    return true;
}

bool erodeRegion( const PointCloud& pointCloud, VertBitSet& region, float erosion, ProgressCallback cb, const AffineXf3f* xf )
{
    auto regionCopy = region;

    const auto res =  BitSetParallelForAll( region, [&] ( VertId testVertex )
    {
        if ( !regionCopy.test( testVertex ) )
            return;

        const Vector3f point = xf ? ( *xf )( pointCloud.points[testVertex] ) : pointCloud.points[testVertex];

        findPointsInBall( pointCloud, point, erosion, [&] ( VertId v, const Vector3f& )
        {
            if ( !regionCopy.test( testVertex ) )
                return;

            if ( !region.test( v ) )
                regionCopy.set( testVertex, false );
        }, xf );
    }, cb );

    if ( !res )
        return false;

    region = regionCopy;
    return true;
}
}
