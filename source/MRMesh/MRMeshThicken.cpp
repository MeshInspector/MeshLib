#include "MRMeshThicken.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRMapOrHashMap.h"
#include "MRMapEdge.h"
#include "MRTimer.h"
#include "MRMeshFillHole.h"

namespace MR
{

Mesh makeThickMesh( const Mesh & m, float halfWidth )
{
    MR_TIMER;
    Mesh res = m;

    auto src2flippedEdges = WholeEdgeMapOrHashMap::createMap();
    PartMapping map{ .src2tgtEdges = &src2flippedEdges };
    res.addMeshPart( m, true, {}, {}, map ); // true = with flipping

    BitSetParallelFor( res.topology.getValidVerts(), [&]( VertId v )
    {
        const auto n = res.pseudonormal( v );
        res.points[v] += halfWidth * n;
    } );

    // stitches(build cylinder) corresponding boundaries of two parts
    StitchHolesParams stitchParams;
    stitchParams.metric = getMinAreaMetric( res );
    for ( EdgeId e : m.topology.findHoleRepresentiveEdges() )
    {
        auto fe = mapEdge( src2flippedEdges, e );
        if ( !fe )
        {
            assert( false );
            continue;
        }
        buildCylinderBetweenTwoHoles( res, e, fe.sym(), stitchParams );
    }

    return res;
}

} //namespace MR
