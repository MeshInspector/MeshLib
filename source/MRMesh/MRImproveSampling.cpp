#include "MRImproveSampling.h"
#include "MRPointCloud.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRTimer.h"

namespace MR
{

bool improveSampling( const PointCloud & cloud, VertBitSet & samples, int numIters, const ProgressCallback & progress )
{
    MR_TIMER
    assert( numIters >= 1 );

    // create point-cloud from initial samples
    PointCloud cloudOfSamples;
    cloudOfSamples.addPartByMask( cloud, samples );

    if ( !reportProgress( progress, 0.1f ) )
        return false;

    VertMap pt2sm( cloud.points.size() );
    const auto sampleSz = cloudOfSamples.points.size();
    Vector<Vector3f, VertId> sumPos( sampleSz );
    Vector<int, VertId> cnt( sampleSz );

    for ( int i = 0; i < numIters; ++i )
    {
        auto cb = subprogress( subprogress( progress, 0.1f, 0.9f ), float( i ) / numIters, float( i + 1 ) / numIters );

        // find the closest sample for each point of the cloud
        if ( !BitSetParallelFor( cloud.validPoints, [&]( VertId v )
            {
                pt2sm[v] = findProjectionOnPoints( cloud.points[v], cloudOfSamples ).vId;
            }, subprogress( cb, 0.1f, 0.6f ) ) )
            return false;

        // find sum of points attributed to each initial sample
        for ( auto sm = 0_v; sm < sampleSz; ++sm )
        {
            sumPos[sm] = {};
            cnt[sm] = 0;
        }
        for ( auto v : cloud.validPoints )
        {
            auto sm = pt2sm[v];
            sumPos[sm] += cloud.points[v];
            ++cnt[sm];
        }

        if ( !reportProgress( cb, 0.7f ) )
            return false;

        // move samples in the average points
        if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId sm )
            {
                if( cnt[sm] <= 0 )
                    return; // e.g. two coinciding points
                cloudOfSamples.points[sm] = sumPos[sm] / float( cnt[sm] );
            }, subprogress( cb, 0.7f, 1.0f ) ) )
            return false;
        cloudOfSamples.invalidateCaches();
    }

    // find points closest to moved samples
    VertMap sm2pt( sampleSz );
    if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId sm )
        {
            sm2pt[sm] = findProjectionOnPoints( cloudOfSamples.points[sm], cloud ).vId;
        }, subprogress( progress, 0.9f, 0.99f ) ) )
        return false;

    // produce new samples
    samples.clear();
    samples.resize( cloud.points.size() );
    for ( auto v : sm2pt )
        samples.set( v );
    return true;
}

} //namespace MR
