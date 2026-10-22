#include "MRRigidXfRansac.h"
#include "MRPointToPointAligningTransform.h"
#include "MRParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"
#include <algorithm>
#include <cmath>
#include <optional>

namespace MR
{

namespace
{

std::uint64_t splitMix( std::uint64_t x )
{
    x += 0x9e3779b97f4a7c15ULL;
    x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
    x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebULL;
    return x ^ ( x >> 31 );
}

} // anonymous namespace

Expected<AffineXf3f> findRigidXfRansac( const VertCoords & srcPoints, const VertCoords & tgtPoints,
    const std::vector<VertPair> & matches, const RigidXfRansacParams & params, const ProgressCallback & progress )
{
    MR_TIMER;
    const auto numMatches = matches.size();
    if ( numMatches < 3 )
        return unexpected( "At least three matches are required" );

    const auto distSq = sqr( params.inlierDistance );
    auto countInliers = [&]( const AffineXf3f & xf )
    {
        int res = 0;
        for ( const auto & m : matches )
            if ( distanceSq( xf( srcPoints[m.first] ), tgtPoints[m.second] ) <= distSq )
                ++res;
        return res;
    };

    // the transformation of the sample with given index, or nothing if the sample is inconsistent
    auto sampleXf = [&]( int iteration ) -> std::optional<AffineXf3f>
    {
        size_t ids[3];
        auto state = ( std::uint64_t( params.seed ) << 32 ) + std::uint64_t( iteration );
        for ( int k = 0; k < 3; )
        {
            state = splitMix( state );
            ids[k] = size_t( state % numMatches );
            if ( std::find( ids, ids + k, ids[k] ) == ids + k )
                ++k;
        }
        for ( int a = 0; a < 3; ++a )
        {
            const auto & ma = matches[ids[a]];
            const auto & mb = matches[ids[( a + 1 ) % 3]];
            const auto ds = distance( srcPoints[ma.first], srcPoints[mb.first] );
            const auto dt = distance( tgtPoints[ma.second], tgtPoints[mb.second] );
            if ( !( ds > 0 ) || !( std::min( ds, dt ) >= params.edgeLengthRatio * std::max( ds, dt ) ) )
                return {};
        }
        PointToPointAligningTransform p2p;
        for ( auto i : ids )
            p2p.add( srcPoints[matches[i].first], tgtPoints[matches[i].second] );
        return AffineXf3f( p2p.findBestRigidXf() );
    };

    // samples are processed in batches, and the best one is taken from each batch in the order of indices,
    // so the result does not depend on the number of threads
    constexpr int batchSize = 1024;
    std::vector<int> counts( batchSize );
    std::vector<AffineXf3f> xfs( batchSize );
    int bestCount = 0;
    AffineXf3f bestXf;
    int required = params.maxIterations;
    for ( int done = 0; done < required; )
    {
        const auto num = std::min( batchSize, required - done );
        ParallelFor( 0, num, [&]( int i )
        {
            counts[i] = 0;
            if ( auto xf = sampleXf( done + i ) )
            {
                xfs[i] = *xf;
                counts[i] = countInliers( *xf );
            }
        } );
        done += num;
        for ( int i = 0; i < num; ++i )
        {
            if ( counts[i] > bestCount )
            {
                bestCount = counts[i];
                bestXf = xfs[i];
            }
        }

        // the number of samples to have at least one sample of three inliers with given confidence
        const auto inlierSampleProb = std::pow( double( bestCount ) / double( numMatches ), 3.0 );
        if ( inlierSampleProb >= 1 )
            break;
        if ( inlierSampleProb > 0 )
        {
            const auto needed = std::ceil( std::log( 1.0 - params.confidence ) / std::log( 1.0 - inlierSampleProb ) );
            required = int( std::min( needed, double( params.maxIterations ) ) );
        }
        if ( !reportProgress( progress, float( done ) / float( std::max( done, required ) ) ) )
            return unexpectedOperationCanceled();
    }
    if ( bestCount == 0 )
        return unexpected( "No consistent sample of three matches was found" );

    PointToPointAligningTransform p2p;
    for ( const auto & m : matches )
        if ( distanceSq( bestXf( srcPoints[m.first] ), tgtPoints[m.second] ) <= distSq )
            p2p.add( srcPoints[m.first], tgtPoints[m.second] );
    if ( const auto refinedXf = AffineXf3f( p2p.findBestRigidXf() ); countInliers( refinedXf ) >= bestCount )
        return refinedXf;
    return bestXf;
}

} //namespace MR
