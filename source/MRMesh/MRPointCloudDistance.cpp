#include "MRPointCloudDistance.h"
#include "MRPointCloud.h"
#include "MRPointsProject.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

float findMaxDistanceSqOneWay( const PointCloud& a, const PointCloud& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER

    return tbb::parallel_reduce
    (
        tbb::blocked_range( 0_v, b.validPoints.endId() ),
        0.0f,
        [&] ( const tbb::blocked_range<VertId>& range, float init )
        {
            for ( VertId i = range.begin(); i < range.end(); ++i )
            {
                if ( !b.validPoints.test( i ) )
                    continue;

                auto distSq = findProjectionOnPoints( rigidB2A ? (*rigidB2A)( b.points[i] ) : b.points[i], a, maxDistanceSq ).distSq;
                if ( distSq > init )
                    init = distSq;
            }           

            return  init;
        },
        [] ( float a, float b ) -> float
        {
            return a > b ? a : b;
        }
    );
}

float findMaxDistanceSq( const PointCloud& a, const PointCloud& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::unique_ptr<AffineXf3f> rigidA2B = rigidB2A ? std::make_unique<AffineXf3f>( rigidB2A->inverse() ) : nullptr;
    return std::max( findMaxDistanceSqOneWay( a, b, rigidB2A, maxDistanceSq ), findMaxDistanceSqOneWay( b, a, rigidA2B.get(), maxDistanceSq ) );
}

} //namespace MR
