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

void relax( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, SimpleProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return;

    MR_TIMER

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    for ( int i = 0; i < params.iterations; ++i )
    {
        SimpleProgressCallback internalCb;
        if ( cb )
            internalCb = [&] ( float p )
        {
            cb( ( float( i ) + p ) / float( params.iterations ) );
        };
        newPoints = pointCloud.points;
        BitSetParallelFor( zone, [&] ( VertId v )
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
            auto& np = newPoints[v];
            auto pushForce = params.force * ( Vector3f{ sumPos / double( count ) } - np );
            np += pushForce;
        }, internalCb );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

void relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */, SimpleProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return;

    MR_TIMER

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    std::vector<Vector3f> vertPushForces( zone.size() );
    std::vector<std::vector<VertId>> neighbors( zone.size() );
    for ( int i = 0; i < params.iterations; ++i )
    {
        SimpleProgressCallback internalCb1, internalCb2;
        if ( cb )
        {
            internalCb1 = [&] ( float p )
            {
                cb( ( float( i ) + p * 0.5f ) / float( params.iterations ) );
            };
            internalCb2 = [&] ( float p )
            {
                cb( ( float( i ) + p * 0.5f ) / float( params.iterations ) );
            };
        }
        newPoints = pointCloud.points;
        BitSetParallelFor( zone, [&] ( VertId v )
        {
            Vector3d sumPos;
            auto& neighs = neighbors[v];
            neighs.clear();
            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId newV, const Vector3f& position )
            {
                if ( newV != v )
                {
                    neighs.push_back( newV );
                    sumPos += Vector3d( position );
                }
            } );
            if ( neighs.empty() )
                return;
            vertPushForces[v] = params.force * ( Vector3f{ sumPos / double( neighs.size() ) } - pointCloud.points[v] );
        }, internalCb1 );
        BitSetParallelFor( zone, [&] ( VertId v )
        {
            auto& np = newPoints[v];
            np += vertPushForces[v];
            auto modifier = 1.0f / float( neighbors.size() );
            for ( const auto& nv : neighbors[v] )
            {
                if ( zone.test( nv ) )
                    np -= ( vertPushForces[nv] * modifier );
            }
        }, internalCb2 );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

void relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params /*= {} */, SimpleProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return;

    MR_TIMER;

    VertCoords newPoints;
    const VertBitSet& zone = params.region ? *params.region : pointCloud.validPoints;
    if ( !zone.any() )
        return;
    float radius = params.neighborhoodRadius > 0.0f ? params.neighborhoodRadius :
        pointCloud.getBoundingBox().diagonal() * 0.1f;

    bool hasNormals = pointCloud.normals.size() > size_t( pointCloud.validPoints.find_last() );

    for ( int i = 0; i < params.iterations; ++i )
    {
        SimpleProgressCallback internalCb;
        if ( cb )
            internalCb = [&] ( float p )
        {
            cb( ( float( i ) + p ) / float( params.iterations ) );
        };
        newPoints = pointCloud.points;
        BitSetParallelFor( zone, [&] ( VertId v )
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

            auto& np = newPoints[v];
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
        }, internalCb );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

}