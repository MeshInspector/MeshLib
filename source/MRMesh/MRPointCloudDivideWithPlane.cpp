#include "MRPointCloudDivideWithPlane.h"

#include "MRTimer.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRPlane3.h"

namespace MR
{

VertBitSet findHalfSpacePoints( const PointCloud& pc, const Plane3f& plane )
{
    MR_TIMER
    VertBitSet result( pc.validPoints.find_last() + 1 );
    BitSetParallelFor( pc.validPoints, [&] ( VertId v )
    {
        result.set( v, plane.distance( pc.points[v] ) > 0 );
    } );
    return result;
}

PointCloud divideWithPlane( const PointCloud& pc, const Plane3f& plane, const DividePointCloudOptionalOutput& optOut )
{
    MR_TIMER
    const auto posVerts = findHalfSpacePoints( pc, plane );
    PointCloud res;
    res.addPartByMask( pc, posVerts, { .src2tgtVerts = optOut.outVmap } );
    if ( optOut.otherPart )
    {
        *optOut.otherPart = PointCloud{};
        optOut.otherPart->addPartByMask( pc, pc.validPoints - posVerts, { .src2tgtVerts = optOut.otherOutVmap } );
    }
    return res;
}

}