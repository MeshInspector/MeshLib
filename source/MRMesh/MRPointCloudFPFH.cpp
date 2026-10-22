#include "MRPointCloudFPFH.h"
#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRPointsInBall.h"
#include "MRBitSetParallelFor.h"
#include "MRConstants.h"
#include "MRTimer.h"
#include <algorithm>
#include <cfloat>
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

float distanceSq( const FPFHDescriptor & a, const FPFHDescriptor & b )
{
    float res = 0;
    for ( int i = 0; i < FPFHDescriptorSize; ++i )
        res += sqr( a[i] - b[i] );
    return res;
}

// for every point in `from` finds the point in `to` with the closest descriptor
bool findClosestDescriptors( const FPFHDescriptors & fromDescriptors, const VertBitSet & from,
    const FPFHDescriptors & toDescriptors, const VertBitSet & to, VertMap & closest, const ProgressCallback & progress )
{
    // contiguous copy of the target descriptors for cache-friendly scanning
    std::vector<VertId> toIds;
    toIds.reserve( to.count() );
    std::vector<FPFHDescriptor> toDescs;
    toDescs.reserve( toIds.capacity() );
    for ( auto v : to )
    {
        toIds.push_back( v );
        toDescs.push_back( toDescriptors[v] );
    }

    closest.resize( from.find_last() + 1 );
    return BitSetParallelFor( from, [&]( VertId v )
    {
        const auto & d = fromDescriptors[v];
        auto bestDistSq = FLT_MAX;
        size_t best = 0;
        for ( size_t i = 0; i < toDescs.size(); ++i )
        {
            if ( const auto distSq = distanceSq( d, toDescs[i] ); distSq < bestDistSq )
            {
                bestDistSq = distSq;
                best = i;
            }
        }
        closest[v] = toIds[best];
    }, progress );
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
    // sigma = 0.2 * radius gave more precise correct matches than other sigmas and 1/distance weights on real scans
    const auto expCoef = -1 / ( 2 * sqr( 0.2f * radius ) );
    FPFHDescriptors res( points.size() );
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        auto & h = res[v];
        findPointsInBall( pointCloud, { points[v], radiusSq }, [&]( const PointsProjectionResult & found, const Vector3f &, Ball3f & )
        {
            const auto weight = std::exp( expCoef * found.distSq );
            const auto & nh = spfh[found.vId];
            for ( int i = 0; i < FPFHDescriptorSize; ++i )
                h[i] += weight * nh[i];
            return Processing::Continue;
        } );
        normalize( h );
    }, fpfhProgress ) )
        return unexpectedOperationCanceled();

    return res;
}

VertBitSet findUniqueFPFHPoints( const FPFHDescriptors & descriptors, const VertBitSet & points, float beta )
{
    MR_TIMER;
    const auto num = points.count();
    if ( num == 0 )
        return {};

    std::array<double, FPFHDescriptorSize> mean{};
    for ( auto v : points )
        for ( int i = 0; i < FPFHDescriptorSize; ++i )
            mean[i] += descriptors[v][i];
    for ( auto & x : mean )
        x /= double( num );

    VertScalars dist( points.find_last() + 1 );
    BitSetParallelFor( points, [&]( VertId v )
    {
        double sum = 0;
        for ( int i = 0; i < FPFHDescriptorSize; ++i )
            sum += sqr( descriptors[v][i] - mean[i] );
        dist[v] = float( std::sqrt( sum ) );
    } );

    double sum = 0, sumSq = 0;
    for ( auto v : points )
    {
        sum += dist[v];
        sumSq += sqr( double( dist[v] ) );
    }
    const auto mu = sum / double( num );
    const auto sigma = std::sqrt( std::max( 0.0, sumSq / double( num ) - sqr( mu ) ) );

    // the floor ignores float rounding in the distances of equal descriptors
    const auto threshold = std::max( beta * sigma, 1e-6 );
    VertBitSet res( dist.size() );
    BitSetParallelFor( points, [&]( VertId v )
    {
        if ( std::abs( dist[v] - mu ) > threshold )
            res.set( v );
    } );
    return res;
}

Expected<VertBitSet> findUniqueFPFHPoints( const PointCloud& pointCloud, float radius, float beta, const ProgressCallback & progress )
{
    MR_TIMER;
    auto descriptors = computeFPFH( pointCloud, radius, progress );
    if ( !descriptors )
        return unexpected( std::move( descriptors.error() ) );
    return findUniqueFPFHPoints( *descriptors, pointCloud.validPoints, beta );
}

Expected<std::vector<VertPair>> findFPFHMatches(
    const FPFHDescriptors & srcDescriptors, const VertBitSet & srcPoints,
    const FPFHDescriptors & tgtDescriptors, const VertBitSet & tgtPoints,
    bool mutual, const ProgressCallback & progress )
{
    MR_TIMER;
    std::vector<VertPair> res;
    if ( srcPoints.none() || tgtPoints.none() )
        return res;

    auto [srcProgress, tgtProgress] = splitProgress( progress, mutual ? 0.5f : 1.0f );
    VertMap src2tgt;
    if ( !findClosestDescriptors( srcDescriptors, srcPoints, tgtDescriptors, tgtPoints, src2tgt, srcProgress ) )
        return unexpectedOperationCanceled();

    VertMap tgt2src;
    if ( mutual && !findClosestDescriptors( tgtDescriptors, tgtPoints, srcDescriptors, srcPoints, tgt2src, tgtProgress ) )
        return unexpectedOperationCanceled();

    for ( auto v : srcPoints )
        if ( !mutual || tgt2src[src2tgt[v]] == v )
            res.emplace_back( v, src2tgt[v] );
    return res;
}

} //namespace MR
