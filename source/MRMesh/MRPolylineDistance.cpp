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

PolylinePolylineDistanceResult findMaxDistanceOneWay( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    MR_TIMER;

    PolylinePolylineDistanceResult res;
    const auto& validVerts = b.topology.getValidVerts();
    const auto firstVert = validVerts.find_first();
    if ( !firstVert )
        return res;

    if ( !a.topology.lastNotLoneEdge().valid() )
    {
        // nothing to measure the distance to
        res.b = b.toEdgePoint( firstVert );
        res.distSq = maxDistanceSq;
        return res;
    }

    auto inA = [&]( VertId v ) { return rigidB2A ? ( *rigidB2A )( b.points[v] ) : b.points[v]; };

    struct VertDist
    {
        VertId v;
        float distSq = -1;
    };
    a.getAABBTree(); // prepare the tree before the parallel region
    // of several equally distant vertices, the one with the smallest id is taken
    const auto best = tbb::parallel_reduce
    (
        tbb::blocked_range( firstVert, validVerts.find_last() + 1 ),
        VertDist{},
        [&] ( const tbb::blocked_range<VertId>& range, VertDist init )
        {
            for ( VertId i = range.begin(); i < range.end(); ++i )
            {
                if ( !validVerts.test( i ) )
                    continue;
                const auto distSq = findProjectionOnPolyline( inA( i ), a, maxDistanceSq ).distSq;
                if ( distSq > init.distSq )
                    init = { i, distSq };
            }
            return init;
        },
        [] ( const VertDist& x, const VertDist& y ) { return y.distSq > x.distSq ? y : x; }
    );

    const auto proj = findProjectionOnPolyline( inA( best.v ), a, maxDistanceSq );
    if ( proj.valid() )
        res.a = a.toEdgePoint( EdgeId( proj.line ), proj.point );
    res.b = b.toEdgePoint( best.v );
    res.distSq = proj.distSq;
    return res;
}

PolylinePolylineDistanceResult findMaxDistance( const Polyline3& a, const Polyline3& b, const AffineXf3f* rigidB2A, float maxDistanceSq )
{
    std::optional<AffineXf3f> rigidA2B;
    if ( rigidB2A )
        rigidA2B = rigidB2A->inverse();
    auto res = findMaxDistanceOneWay( a, b, rigidB2A, maxDistanceSq );
    const auto ba = findMaxDistanceOneWay( b, a, rigidA2B ? &*rigidA2B : nullptr, maxDistanceSq );
    if ( ba.distSq > res.distSq )
        res = { ba.b, ba.a, ba.distSq };
    return res;
}

} //namespace MR
