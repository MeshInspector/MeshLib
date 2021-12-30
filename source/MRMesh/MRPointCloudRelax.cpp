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

void relax( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */ )
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
        } );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

void relaxKeepVolume( PointCloud& pointCloud, const PointCloudRelaxParams& params /*= {} */ )
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
        } );
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
        } );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

void relaxApprox( PointCloud& pointCloud, const PointCloudApproxRelaxParams& params /*= {} */ )
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

    for ( int i = 0; i < params.iterations; ++i )
    {
        newPoints = pointCloud.points;
        BitSetParallelFor( zone, [&] ( VertId v )
        {
            PointAccumulator accum;
            std::vector<VertId> neighbors;
            Vector3d centroid;
            findPointsInBall( pointCloud, pointCloud.points[v], radius,
                [&] ( VertId newV, const Vector3f& position )
            {
                Vector3d ptD = Vector3d( position );
                centroid += ptD;
                accum.addPoint( ptD );
                neighbors.push_back( newV );
            } );
            if ( neighbors.size() < 6 )
                return;

            centroid /= double( neighbors.size() );
            auto& np = newPoints[v];
            Vector3f target;
            auto plane = accum.getBestPlane();
            if ( params.type == RelaxApproxType::Planar )
            {
                target = Plane3f( plane ).project( np );
            }
            else if ( params.type == RelaxApproxType::Quadric )
            {
                AffineXf3d basis;
                basis.A.z = plane.n.normalized();
                auto [x, y] = basis.A.z.perpendicular();
                basis.A.x = x;
                basis.A.y = y;
                basis.A = basis.A.transposed();
                basis.b = Vector3d( np );
                auto basisInv = basis.inverse();
                QuadricApprox approxAccum;
                for ( auto newV : neighbors )
                    approxAccum.addPoint( basisInv( Vector3d( pointCloud.points[newV] ) ) );
                auto res = QuadricApprox::findZeroProjection( approxAccum.calcBestCoefficients() );
                target = Vector3f( basis( res ) );
            }
            np += ( params.force * ( 0.5f * target + Vector3f( 0.5 * centroid ) - np ) );
        } );
        pointCloud.points.swap( newPoints );
        pointCloud.invalidateCaches();
    }
}

}