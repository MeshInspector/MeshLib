#include "MRPointCloudRelax.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsInBall.h"
#include "MRBox.h"
#include "MRBestFit.h"
#include "MRBestFitQuadric.h"
#include "MRVector4.h"

namespace MR
{

bool relax( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        ProgressCallback internalCb;
        if ( cb )
        {
            internalCb = [&] ( float p )
            {
                return cb( ( float( i ) + p ) / float( params.iterations ) );
            };
        }
        newPoints = pointCloud.points;
        keepGoing = BitSetParallelFor( zone, [&] ( VertId v )
        {
            Vector3d sumPos;
            int count = 0;
            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId newV, const Vector3f& position )
            {
                if ( newV != v )
                {
                    sumPos += Vector3d( position );
                    count++;
                }
            } );
            if ( count == 0 )
                return;
            auto np = newPoints[v];
            auto pushForce = params.force * ( Vector3f{ sumPos / double( count ) } - np );
            np += pushForce;
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

bool relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    std::vector<Vector3f> vertPushForces( zone.size() );

    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        ProgressCallback internalCb1, internalCb2;
        if ( cb )
        {
            internalCb1 = [&] ( float p )
            {
                return cb( ( float( i ) + p * 0.5f ) / float( params.iterations ) );
            };
            internalCb2 = [&] ( float p )
            {
                return cb( ( float( i ) + p * 0.5f + 0.5f ) / float( params.iterations ) );
            };
        }
        newPoints = pointCloud.points;
        keepGoing = BitSetParallelFor( zone, [&] ( VertId v )
        {
            Vector3d sumPos;
            int count = 0;
            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId nv, const Vector3f& position )
            {
                if ( nv != v && zone.test( nv ) )
                {
                    sumPos += Vector3d( position );
                    ++count;
                }
            } );
            if ( count <= 0 )
                return;
            vertPushForces[v] = params.force * ( Vector3f{ sumPos / double( count ) } - pointCloud.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;
        keepGoing = BitSetParallelFor( zone, [&] ( VertId v )
        {
            Vector3d sumForces;
            int count = 0;
            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId nv, const Vector3f& )
            {
                if ( nv != v && zone.test( nv ) )
                {
                    sumForces += Vector3d( vertPushForces[nv] );
                    ++count;
                }
            } );
            if ( count <= 0 )
                return;

            auto np = newPoints[v] + vertPushForces[v] - Vector3f{ sumForces / double( count ) };
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb2 );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

bool relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    bool hasNormals = pointCloud.normals.size() > size_t( pointCloud.validPoints.find_last() );
    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        ProgressCallback internalCb;
        if ( cb )
        {
            internalCb = [&] ( float p )
            {
                return cb( ( float( i ) + p ) / float( params.iterations ) );
            };
        }
        newPoints = pointCloud.points;
        keepGoing = BitSetParallelFor( zone, [&] ( VertId v )
        {
            PointAccumulator accum;
            std::vector<std::pair<VertId, double>> weightedNeighbors;

            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId newV, const Vector3f& position )
            {
                double w = 1.0;
                if ( hasNormals )
                    w = dot( pointCloud.normals[v], pointCloud.normals[newV] );
                if ( w > 0.0 )
                {
                    weightedNeighbors.push_back( { newV,w } );
                    accum.addPoint( Vector3d( position ), w );
                }
            } );
            if ( weightedNeighbors.size() < 6 )
                return;

            auto np = newPoints[v];
            Vector3f target;
            if ( params.type == RelaxApproxType::Planar )
                target = accum.getBestPlanef().project( np );
            else if ( params.type == RelaxApproxType::Quadric )
            {
                AffineXf3d basis = accum.getBasicXf();
                basis.A = basis.A.transposed();
                std::swap( basis.A.x, basis.A.y );
                std::swap( basis.A.y, basis.A.z );
                basis.A = basis.A.transposed();
                auto basisInv = basis.inverse();

                QuadricApprox approxAccum;
                for ( auto [newV, w] : weightedNeighbors )
                    approxAccum.addPoint( basisInv( Vector3d( pointCloud.points[newV] ) ), w );

                auto centerPoint = basisInv( Vector3d( pointCloud.points[v] ) );
                const auto coefs = approxAccum.calcBestCoefficients();
                centerPoint.z =
                    coefs[0] * centerPoint.x * centerPoint.x +
                    coefs[1] * centerPoint.x * centerPoint.y +
                    coefs[2] * centerPoint.y * centerPoint.y +
                    coefs[3] * centerPoint.x +
                    coefs[4] * centerPoint.y +
                    coefs[5];
                target = Vector3f( basis( centerPoint ) );
            }
            np += ( params.force * ( target - np ) );
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
        if ( !keepGoing )
            break;
    }
    return keepGoing;
}

} //namespace MR
