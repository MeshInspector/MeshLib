#include "MRImproveSampling.h"
#include "MRPointCloud.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRVector4.h"
#include "MRColor.h"
#include "MRTimer.h"

namespace MR
{

bool improveSampling( const PointCloud & cloud, VertBitSet & samples, const ImproveSamplingSettings & settings )
{
    MR_TIMER
    assert( settings.numIters >= 1 );

    // create point-cloud from initial samples
    PointCloud cloudOfSamples;
    cloudOfSamples.addPartByMask( cloud, samples );

    if ( !reportProgress( settings.progress, 0.1f ) )
        return false;

    VertMap pt2sm( cloud.points.size() );
    const auto sampleSz = cloudOfSamples.points.size();
    Vector<Vector3f, VertId> sumPos( sampleSz );
    Vector<int, VertId> ptsInSm( sampleSz );

    for ( int i = 0; i < settings.numIters; ++i )
    {
        auto cb = subprogress( subprogress( settings.progress, 0.1f, 0.9f ),
            float( i ) / settings.numIters, float( i + 1 ) / settings.numIters );

        // find the closest sample for each point of the cloud
        if ( !BitSetParallelFor( cloud.validPoints, [&]( VertId v )
            {
                pt2sm[v] = findProjectionOnPoints( cloud.points[v], cloudOfSamples ).vId;
            }, subprogress( cb, 0.1f, 0.6f ) ) )
            return false;

        // find sum of points attributed to each initial sample
        sumPos.clear();
        sumPos.resize( sampleSz );
        ptsInSm.clear();
        ptsInSm.resize( sampleSz );
        for ( auto v : cloud.validPoints )
        {
            auto s = pt2sm[v];
            sumPos[s] += cloud.points[v];
            ++ptsInSm[s];
        }

        if ( !reportProgress( cb, 0.7f ) )
            return false;

        // move samples in the average points
        if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId s )
            {
                if( ptsInSm[s] <= 0 )
                    return; // e.g. two coinciding points
                cloudOfSamples.points[s] = sumPos[s] / float( ptsInSm[s] );
            }, subprogress( cb, 0.7f, 1.0f ) ) )
            return false;
        cloudOfSamples.invalidateCaches();
    }

    // find points closest to moved samples
    VertMap sm2pt( sampleSz );
    if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId s )
        {
            sm2pt[s] = findProjectionOnPoints( cloudOfSamples.points[s], cloud ).vId;
        }, subprogress( settings.progress, 0.9f, 0.99f ) ) )
        return false;

    // produce new samples
    samples.clear();
    samples.resize( cloud.points.size() );
    for ( VertId i = 0_v; i < sm2pt.size(); ++i )
    {
        if ( ptsInSm[i] < settings.minPointsInSample )
            continue;
        samples.set( sm2pt[i] );
    }

    if ( settings.pt2sm )
    {
        BitSetParallelFor( cloud.validPoints, [&]( VertId v )
        {
            auto s = pt2sm[v];
            if ( ptsInSm[s] < settings.minPointsInSample )
                pt2sm[v] = {};
        } );
        *settings.pt2sm = std::move( pt2sm );
    }

    if ( settings.cloudOfSamples )
    {
        BitSetParallelFor( cloudOfSamples.validPoints, [&]( VertId s )
        {
            if ( ptsInSm[s] < settings.minPointsInSample )
                cloudOfSamples.validPoints.reset( s );
        } );

        if ( cloud.hasNormals() )
        {
            sumPos.clear();
            sumPos.resize( sampleSz );
            for ( auto v : cloud.validPoints )
            {
                auto s = pt2sm[v];
                sumPos[s] += cloud.normals[v];
            }
            cloudOfSamples.normals.resizeNoInit( sampleSz );
            BitSetParallelFor( cloudOfSamples.validPoints, [&]( VertId s )
            {
                cloudOfSamples.normals[s] = sumPos[s].normalized();
            } );
        }

        *settings.cloudOfSamples = std::move( cloudOfSamples );
    }

    if ( settings.ptColors && settings.smColors )
    {
        Vector<Vector4f, VertId> sumCol( sampleSz );
        sumCol.resize( sampleSz );
        for ( auto v : cloud.validPoints )
        {
            auto s = pt2sm[v];
            sumCol[s] += Vector4f( (*settings.ptColors)[v] );
        }
        settings.smColors->resizeNoInit( sampleSz );
        // cloudOfSamples may be already moved before
        ParallelFor( sumCol, [&]( VertId s )
        {
            if ( ptsInSm[s] > 0 )
                (*settings.smColors)[s] = Color( sumCol[s] / float( ptsInSm[s] ) );
        } );
    }

    if ( settings.ptsInSm )
    {
        ParallelFor( ptsInSm, [&]( VertId s )
        {
            if ( ptsInSm[s] < settings.minPointsInSample )
                ptsInSm[s] = 0;
        } );
        *settings.ptsInSm = std::move( ptsInSm );
    }

    return true;
}

} //namespace MR
