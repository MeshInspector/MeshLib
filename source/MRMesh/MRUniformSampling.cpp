#include "MRUniformSampling.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRBox.h"
#include <cfloat>

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

std::optional<VertBitSet> pointRegularUniformSampling( const PointCloud& pointCloud, float distance, 
    const ProgressCallback& cb /*= {} */ )
{
    MR_TIMER

    std::vector<VertId> searchQueue = pointCloud.getLexicographicalOrder();

    if ( cb && !cb( 0.3f ) )
        return {};

    int progressCounter = 0;
    auto sp = subprogress( cb, 0.3f, 1.0f );

    VertBitSet visited( pointCloud.validPoints.size() );
    VertBitSet sampled( pointCloud.validPoints.size() );
    for ( auto v : searchQueue )
    {
        if ( sp && !( ( ++progressCounter ) & 0x3ff ) &&
            !sp( float( progressCounter ) / float( searchQueue.size() ) ) )
            return {};
        if ( visited.test( v ) )
            continue;
        sampled.set( v );
        findPointsInBall( pointCloud, pointCloud.points[v], distance, [&] ( VertId cv, const Vector3f& )
        {
            visited.set( cv );
        } );
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
    res->addPartByMask( pointCloud, *optVerts, {}, extNormals );

    if ( !reportProgress( cb, 1.0f ) )
        res.reset();
    return res;
}

} //namespace MR
