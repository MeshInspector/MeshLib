#include "MRPolylineDistance.h"
#include "MRPolyline.h"
#include "MRPolylineProject.h"
#include "MRAffineXf3.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <optional>

namespace MR
{

float findMaxDistanceSqOneWay( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    const auto& validVerts = b.topology.getValidVerts();
    if ( !a.topology.lastNotLoneEdge().valid() )
        return validVerts.any() ? maxDistanceSq : 0.0f; // nothing to measure the distance to

    a.getAABBTree(); // prepare the tree before the parallel region
    return tbb::parallel_reduce
    (
        tbb::blocked_range( 0_v, validVerts.endId() ),
        0.0f,
        [&] ( const tbb::blocked_range<VertId>& range, float init )
        {
            for ( VertId i = range.begin(); i < range.end(); ++i )
            {
                if ( !validVerts.test( i ) )
                    continue;

                auto distSq = findProjectionOnPolyline( rigidB2A ? (*rigidB2A)( b.points[i] ) : b.points[i], a, maxDistanceSq ).distSq;
                if ( distSq > init )
                    init = distSq;
            }

            return init;
        },
        [] ( float a, float b ) -> float
        {
            return a > b ? a : b;
        }
    );
}

float findMaxDistanceSq( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::optional<AffineXf3f> rigidA2B;
    if ( rigidB2A )
        rigidA2B = rigidB2A->inverse();
    return std::max( findMaxDistanceSqOneWay( a, b, rigidB2A, maxDistanceSq ), findMaxDistanceSqOneWay( b, a, rigidA2B ? &*rigidA2B : nullptr, maxDistanceSq ) );
}

} //namespace MR
