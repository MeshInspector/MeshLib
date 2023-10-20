#include "MRImproveSampling.h"
#include "MRPointCloud.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRTimer.h"

namespace MR
{

std::optional<VertBitSet> improveSampling( const PointCloud & cloud, const VertBitSet & iniSamples, const ProgressCallback & cb )
{
    MR_TIMER
    std::optional<VertBitSet> res;

    // create point-cloud from initial samples
    PointCloud iniSampledCloud;
    iniSampledCloud.addPartByMask( cloud, iniSamples );

    if ( !reportProgress( cb, 0.1f ) )
        return res;

    // find the closest initial sample for each point of the cloud
    VertMap pt2sm( cloud.points.size() );
    if ( !BitSetParallelFor( cloud.validPoints, [&]( VertId v )
        {
            pt2sm[v] = findProjectionOnPoints( cloud.points[v], iniSampledCloud ).vId;
        }, subprogress( cb, 0.1f, 0.6f ) ) )
        return res;

    // find sum of points attributed to each initial sample
    const auto iniSz = iniSampledCloud.points.size();
    Vector<Vector3f, VertId> sumPos( iniSz );
    Vector<int, VertId> cnt( iniSz );
    for ( auto v : cloud.validPoints )
    {
        auto sm = pt2sm[v];
        sumPos[sm] += cloud.points[v];
        ++cnt[sm];
    }

    if ( !reportProgress( cb, 0.7f ) )
        return res;

    // find new samples closest to average points
    VertMap sm2pt( iniSz );
    if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId sm )
        {
            if( cnt[sm] <= 0 )
                return; // two coinciding points
            const auto avgPos = sumPos[sm] / float( cnt[sm] );
            sm2pt[sm] = findProjectionOnPoints( avgPos, cloud ).vId;
        }, subprogress( cb, 0.7f, 0.9f ) ) )
        return res;

    // produce new samples
    res.emplace( cloud.points.size() );
    for ( auto v : sm2pt )
        if ( v )
            res->set( v );

    return res;
}

bool improveSampling( const PointCloud & cloud, VertBitSet & samples, int numIters, const ProgressCallback & cb )
{
    MR_TIMER
    assert( numIters >= 1 );

    for ( int i = 0; i < numIters; ++i )
    {
        auto optSamples = improveSampling( cloud, samples, subprogress( cb, float( i ) / numIters, float( i + 1 ) / numIters ) );
        if ( !optSamples )
            return false;
        samples = std::move( *optSamples );
    }

    return true;
}

} //namespace MR
