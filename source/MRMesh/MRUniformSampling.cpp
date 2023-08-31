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

}
