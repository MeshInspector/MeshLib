#include "MRPointCloudFPFH.h"
#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRPointsInBall.h"
#include "MRBitSetParallelFor.h"
#include "MRConstants.h"
#include "MRTimer.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace MR
{

namespace
{

int binIndex( float x, float lo, float hi )
{
    return std::clamp( int( FPFHBins * ( x - lo ) / ( hi - lo ) ), 0, FPFHBins - 1 );
}

// adds the angles between oriented points (p,n) and (q,m) in the histograms, skipping a degenerate pair
void addPair( FPFHDescriptor & h, const Vector3f & p, const Vector3f & n, const Vector3f & q, const Vector3f & m )
{
    auto d = q - p;
    const auto len = d.length();
    if ( !( len > 0 ) )
        return;
    d /= len;

    // the frame is built on the point whose normal is closer to the line between the points, which makes the angles symmetric
    auto u = n;
    auto n2 = m;
    auto phi = dot( u, d );
    if ( const auto phi2 = dot( n2, d ); std::abs( phi ) < std::abs( phi2 ) )
    {
        std::swap( u, n2 );
        d = -d;
        phi = -phi2;
    }
    auto v = cross( d, u );
    const auto vLen = v.length();
    if ( !( vLen > 0 ) )
        return;
    v /= vLen;
    const auto w = cross( u, v );

    const auto alpha = dot( v, n2 );
    const auto theta = std::atan2( dot( w, n2 ), dot( u, n2 ) );
    h[binIndex( theta, -PI_F, PI_F )] += 1;
    h[FPFHBins + binIndex( alpha, -1.0f, 1.0f )] += 1;
    h[2 * FPFHBins + binIndex( phi, -1.0f, 1.0f )] += 1;
}

// makes each of three histograms sum to 1, or leaves it zero
void normalize( FPFHDescriptor & h )
{
    for ( int i = 0; i < 3; ++i )
    {
        const auto first = h.begin() + i * FPFHBins;
        const auto last = first + FPFHBins;
        if ( const auto sum = std::accumulate( first, last, 0.0f ); sum > 0 )
            std::for_each( first, last, [k = 1 / sum]( float & x ) { x *= k; } );
    }
}

} // anonymous namespace

Expected<FPFHDescriptors> computeFPFH( const PointCloud& pointCloud, float radius, const ProgressCallback & progress )
{
    MR_TIMER;
    if ( !pointCloud.hasNormals() )
        return unexpected( "Point cloud has no normals" );

    const auto radiusSq = sqr( radius );
    const auto & points = pointCloud.points;
    const auto & normals = pointCloud.normals;
    auto [spfhProgress, fpfhProgress] = splitProgress( progress, 0.5f );

    // simplified histograms of every point from the angles to its neighbours
    FPFHDescriptors spfh( points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        auto & h = spfh[v];
        findPointsInBall( pointCloud, { points[v], radiusSq }, [&]( const PointsProjectionResult & found, const Vector3f & pos, Ball3f & )
        {
            if ( found.vId != v )
                addPair( h, points[v], normals[v], pos, normals[found.vId] );
            return Processing::Continue;
        } );
        normalize( h );
    }, spfhProgress ) )
        return unexpectedOperationCanceled();

    // final histograms are the averages of the histograms of the point and its neighbours with Gaussian weights of the distance,
    // sigma = 0.4 * radius gave more correct matches than other sigmas and 1/distance weights on real scans
    const auto expCoef = -1 / ( 2 * sqr( 0.4f * radius ) );
    FPFHDescriptors res( points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        auto & h = res[v];
        findPointsInBall( pointCloud, { points[v], radiusSq }, [&]( const PointsProjectionResult & found, const Vector3f &, Ball3f & )
        {
            const auto weight = std::exp( expCoef * found.distSq );
            const auto & nh = spfh[found.vId];
            for ( int i = 0; i < 3 * FPFHBins; ++i )
                h[i] += weight * nh[i];
            return Processing::Continue;
        } );
        normalize( h );
    }, fpfhProgress ) )
        return unexpectedOperationCanceled();

    return res;
}

} //namespace MR
