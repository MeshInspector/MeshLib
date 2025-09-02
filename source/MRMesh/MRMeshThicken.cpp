#include "MRMeshThicken.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"

namespace MR
{

Mesh makeThickMesh( const Mesh & m, float halfWidth )
{
    MR_TIMER;
    Mesh res = m;
    auto holesRepr = m.topology.findHoleRepresentiveEdges();
    EdgeLoops mHoles( holesRepr.size() );
    EdgeLoops extHoles( holesRepr.size() );
    for ( int i = 0; i < holesRepr.size(); ++i )
    {
        mHoles[i] = trackRightBoundaryLoop( m.topology, holesRepr[i] );
        auto e = makeDegenerateBandAroundHole( res, holesRepr[i] );
        extHoles[i] = trackRightBoundaryLoop( res.topology, e );
    }
    res.addMeshPart( m, true, extHoles, mHoles );

    // prepare all normals before modifying the points
    VertNormals ns( res.topology.vertSize() );
    BitSetParallelFor( res.topology.getValidVerts(), [&]( VertId v )
    {
        ns[v] = res.pseudonormal( v ); // degenerate faces will be automatically ignored
    } );

    BitSetParallelFor( res.topology.getValidVerts(), [&]( VertId v )
    {
        res.points[v] += halfWidth * ns[v];
    } );

    return res;
}

} //namespace MR
