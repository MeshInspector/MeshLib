#include "MRImproveSampling.h"
#include "MRPointCloud.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRTimer.h"
#include "MRBestFitQuadric.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"

namespace MR
{

bool improveSampling( const PointCloud & cloud, VertBitSet & samples, int numIters, 
    const ImproveSamplingCurvatureMode& curvatureMode, const ProgressCallback& progress )
{
    MR_TIMER
    assert( numIters >= 1 );

    // create point-cloud from initial samples
    PointCloud cloudOfSamples;
    cloudOfSamples.addPartByMask( cloud, samples );

    if ( !reportProgress( progress, 0.1f ) )
        return false;

    const auto sampleSz = cloudOfSamples.points.size();

    VertMap pt2sm( cloud.points.size() );
    VertScalars pt2w;
    bool useCurv = curvatureMode.radius > 0.0f && curvatureMode.type != ImproveSamplingCurvatureMode::None && sampleSz > 1;
    if ( useCurv )
    {
        pt2w.resize( cloud.points.size() );
        float radius = curvatureMode.radius;
        constexpr float maxWeightExp = 10.0f;
        constexpr float range = 1.0f / 1.5f * ( maxWeightExp - 1.0f );
        tbb::enumerable_thread_specific<std::vector<VertId>> cacheVerts;
        if ( !BitSetParallelFor( cloud.validPoints, [&] ( VertId v )
        {
            auto& lVerts = cacheVerts.local();
            lVerts.clear();
            PointAccumulator planeApprox;
            findPointsInBall( cloud, cloud.points[v], radius, [&] ( VertId cv, const Vector3f& )
            {
                lVerts.push_back( cv );
            } );
            if ( lVerts.size() < 4 )
            {
                pt2w[v] = -FLT_MAX;
                return;
            }
            for ( auto cv : lVerts )
                planeApprox.addPoint( cloud.points[cv] );

            AffineXf3d basis = planeApprox.getBasicXf();
            basis.A = basis.A.transposed();
            std::swap( basis.A.x, basis.A.y );
            std::swap( basis.A.y, basis.A.z );
            basis.A = basis.A.transposed();
            auto basisInv = basis.inverse();

            QuadricApprox qa;
            for ( auto cv : lVerts )
                qa.addPoint( basisInv( Vector3d( cloud.points[cv] ) ) );

            auto coeffs = qa.calcBestCoefficients();
            double sumCoeff = ( std::abs( coeffs[0] ) + std::abs( coeffs[1] ) + std::abs( coeffs[2] ) );

            float w = std::clamp( float( sumCoeff ) * radius, 0.0f, 1.5f );
            pt2w[v] = std::exp( w * range );// +1.0f;
        }, subprogress( progress, 0.1f, 0.2f ) ) )
            return false;
    }

    VertCoords sumPos( sampleSz );
    VertScalars cnt( sampleSz );
        
    for ( int i = 0; i < numIters; ++i )
    {
        auto cb = subprogress( subprogress( progress, useCurv ? 0.2f : 0.1f, 0.9f ),
            float( i ) / numIters, float( i + 1 ) / numIters );

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
            if ( !useCurv )
            {
                cnt[sm] += 1.0f;
                sumPos[sm] += cloud.points[v];
            }
            else
            {
                cnt[sm] += pt2w[v];
                sumPos[sm] += pt2w[v] * cloud.points[v];
            }
        }

        if ( !reportProgress( cb, 0.7f ) )
            return false;

        // move samples in the average points
        if ( !ParallelFor( 0_v, sumPos.endId(), [&]( VertId sm )
            {
                if( cnt[sm] <= 0 )
                    return; // e.g. two coinciding points
                cloudOfSamples.points[sm] = sumPos[sm] / cnt[sm];
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
