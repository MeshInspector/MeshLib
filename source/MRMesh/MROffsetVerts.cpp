#include "MROffsetVerts.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"
#include "MRPositionVertsSmoothly.h"
#include "MRMapOrHashMap.h"

namespace MR
{

bool offsetVerts( Mesh& mesh, const VertMetric& offset, const ProgressCallback& cb )
{
    MR_TIMER;
    mesh.invalidateCaches();

    // prepare all normals before modifying the points
    VertNormals ns( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        ns[v] = mesh.pseudonormal( v );
    } );

    return BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        mesh.points[v] += offset( v ) * ns[v];
    }, cb );
}

Mesh makeThickMesh( const Mesh & m, float halfWidth )
{
    MR_TIMER;

    VertNormals ns( m.topology.vertSize() );
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        ns[v] = m.pseudonormal( v );
    } );
    positionVertsSmoothlySharpBd( m.topology, ns, { .stabilizer = 0.01f } );
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        ns[v] = ns[v].normalized();
    } );

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
    PartMapping map;
    auto m2resVerts = VertMapOrHashMap::createMap();
    map.src2tgtVerts = &m2resVerts;
    res.addMeshPart( m, true, extHoles, mHoles, map );
    ns.resize( res.topology.vertSize() );
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        auto resV = getAt( m2resVerts, v );
        if ( !resV )
        {
            assert( false );
            return;
        }
        ns[resV] = -ns[v];
    } );

    BitSetParallelFor( res.topology.getValidVerts(), [&]( VertId v )
    {
        res.points[v] += halfWidth * ns[v];
    } );

    // degenerate faces will be automatically ignored during pseudonormal computation
    //offsetVerts( res, [halfWidth]( VertId ) { return halfWidth; } );

    return res;
}

} //namespace MR
