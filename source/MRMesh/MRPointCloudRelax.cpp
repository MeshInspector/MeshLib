#include "MRPointCloudRelax.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRNewValuesStorage.hpp"
#include "MRPointsInBall.h"
#include "MRBox.h"
#include "MRBestFit.h"
#include "MRBestFitQuadric.h"
#include "MRVector4.h"

namespace MR
{

static void updateOrInvalidateCaches( PointCloud& pointCloud, const RelaxParams& params )
{
    if ( params.updateCaches )
        pointCloud.updateCaches( params.region );
    else
        pointCloud.invalidateCaches();
}

bool relax( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    NewValuesStorage storage( pointCloud.points, zone );
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
        keepGoing = storage.parallelProcess( [&, radiusSq = sqr( radius )] ( VertId v )
        {
            Vector3d sumPos;
            int count = 0;
            findPointsInBall( pointCloud, { pointCloud.points[v], radiusSq },
                [&] ( const PointsProjectionResult & found, const Vector3f & foundPos, Ball3f & )
            {
                if ( found.vId != v )
                {
                    sumPos += Vector3d( foundPos );
                    count++;
                }
                return Processing::Continue;
            } );
            auto np = pointCloud.points[v];
            if ( count == 0 )
                return np;
            auto pushForce = params.force * ( Vector3f{ sumPos / double( count ) } - np );
            np += pushForce;
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            return np;
        }, internalCb );
        if ( !keepGoing )
            break;
        if ( i + 1 < params.iterations )
            pointCloud.updateCaches( params.region ); // refit is much faster than rebuilding the tree in the next iteration
    }
    updateOrInvalidateCaches( pointCloud, params );
    return keepGoing;
}

bool relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    NewValuesStorage storage( pointCloud.points, zone );
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
        keepGoing = BitSetParallelFor( zone, [&, radiusSq = sqr( radius )] ( VertId v )
        {
            Vector3d sumPos;
            int count = 0;
            findPointsInBall( pointCloud, { pointCloud.points[v], radiusSq },
                [&] ( const PointsProjectionResult & found, const Vector3f & foundPos, Ball3f & )
            {
                if ( found.vId != v && zone.test( found.vId ) )
                {
                    sumPos += Vector3d( foundPos );
                    ++count;
                }
                return Processing::Continue;
            } );
            if ( count <= 0 )
                return;
            vertPushForces[v] = params.force * ( Vector3f{ sumPos / double( count ) } - pointCloud.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;
        keepGoing = storage.parallelProcess( [&, radiusSq = sqr( radius )] ( VertId v )
        {
            Vector3d sumForces;
            int count = 0;
            findPointsInBall( pointCloud, { pointCloud.points[v], radiusSq },
                [&] ( const PointsProjectionResult & found, const Vector3f &, Ball3f & )
            {
                const auto nv = found.vId;
                if ( nv != v && zone.test( nv ) )
                {
                    sumForces += Vector3d( vertPushForces[nv] );
                    ++count;
                }
                return Processing::Continue;
            } );
            if ( count <= 0 )
                return pointCloud.points[v];

            auto np = pointCloud.points[v] + vertPushForces[v] - Vector3f{ sumForces / double( count ) };
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            return np;
        }, internalCb2 );
        if ( !keepGoing )
            break;
        if ( i + 1 < params.iterations )
            pointCloud.updateCaches( params.region ); // refit is much faster than rebuilding the tree in the next iteration
    }
    updateOrInvalidateCaches( pointCloud, params );
    return keepGoing;
}

bool relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params /*= {} */, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = pointCloud.points;

    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return true;
    NewValuesStorage storage( pointCloud.points, zone );
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
        keepGoing = storage.parallelProcess( [&, radiusSq = sqr( radius )] ( VertId v )
        {
            PointAccumulator accum;
            std::vector<std::pair<VertId, double>> weightedNeighbors;

            findPointsInBall( pointCloud, { pointCloud.points[v], radiusSq },
                [&] ( const PointsProjectionResult & found, const Vector3f & foundPos, Ball3f & )
            {
                const auto newV = found.vId;
                double w = 1.0;
                if ( hasNormals )
                    w = dot( pointCloud.normals[v], pointCloud.normals[newV] );
                if ( w > 0.0 )
                {
                    weightedNeighbors.push_back( { newV,w } );
                    accum.addPoint( Vector3d( foundPos ), w );
                }
                return Processing::Continue;
            } );
            auto np = pointCloud.points[v];
            if ( weightedNeighbors.size() < 6 )
                return np;

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
            return np;
        }, internalCb );
        if ( !keepGoing )
            break;
        if ( i + 1 < params.iterations )
            pointCloud.updateCaches( params.region ); // refit is much faster than rebuilding the tree in the next iteration
    }
    updateOrInvalidateCaches( pointCloud, params );
    return keepGoing;
}

} //namespace MR
