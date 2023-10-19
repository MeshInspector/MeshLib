#include "MRUniformSampling.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"

namespace MR
{

std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback & cb )
{
    MR_TIMER

    const auto sz = pointCloud.points.size();
    const int reportStep = std::min( int( sz / 64 ), 1024 );
    VertId reportNext = 0_v;

    VertBitSet res = pointCloud.validPoints;
    for ( auto v : res )
    {
        if ( cb && v >= reportNext )
        {
            if ( !cb( float( v ) / sz ) )
                return {};
            reportNext = v + reportStep;
        }

        findPointsInBall( pointCloud, pointCloud.points[v], distance, [&]( VertId cv, const Vector3f& )
        {
            if ( cv > v )
                res.reset( cv );
        } );
    }
    return res;
}

std::optional<VertBitSet> pointRegularUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback& cb /*= {} */ )
{
    MR_TIMER

    const auto sz = pointCloud.validPoints.count();
    size_t progressCounter = 0;

    auto rp = [&] ()->bool
    {
        if ( !cb )
            return true;
        ++progressCounter;
        if ( bool( progressCounter & 0x3ff ) )
            return true;
        return cb( float( progressCounter ) / float( sz ) );
    };

    VertBitSet visited( pointCloud.validPoints.size() );
    VertBitSet sampled( pointCloud.validPoints.size() );
    for ( auto v : pointCloud.validPoints )
    {
        if ( visited.test( v ) )
        {
            if ( !rp() )
                return {};
            continue;
        }
        visited.set( v );

        VertId nextVertId = v;

        while ( nextVertId )
        {
            sampled.set( nextVertId );
            const auto& nextVertPos = pointCloud.points[nextVertId];
            nextVertId = {};
            const auto maxDistSq = distance * distance;
            float minDistSq = FLT_MAX;
            findPointsInBall( pointCloud, nextVertPos, 2 * distance, [&] ( VertId cv, const Vector3f& pos )
            {
                if ( nextVertId == cv || visited.test( cv ) )
                    return;
                auto distSq = ( nextVertPos - pos ).lengthSq();
                if ( distSq < maxDistSq )
                    visited.set( cv );
                else if ( distSq < minDistSq )
                {
                    minDistSq = distSq;
                    nextVertId = cv;
                }
            } );
            if ( !rp() )
                return {};
        }
    }
    return sampled;
}

std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, float distance, 
    const VertNormals * extNormals, const ProgressCallback & cb )
{
    MR_TIMER

    std::optional<PointCloud> res;
    auto optVerts = pointUniformSampling( pointCloud, distance, subprogress( cb, 0.0f, 0.9f ) );
    if ( !optVerts )
        return res;

    res.emplace();
    res->addPartByMask( pointCloud, *optVerts, nullptr, extNormals );

    if ( !reportProgress( cb, 1.0f ) )
        res.reset();
    return res;
}

} //namespace MR
