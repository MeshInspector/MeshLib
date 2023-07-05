#include "MRInnerShell.h"
#include "MRMesh.h"
#include "MRMeshProject.h"
#include "MRBitSetParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

namespace MR
{

bool isInnerShellVert( const Mesh & mesh, const Vector3f & shellPoint, Side side )
{
    auto sd = findSignedDistance( shellPoint, mesh );
    assert( sd );
    if ( !sd )
        return false;
    if ( sd->mtp.isBd( mesh.topology ) )
        return false;
    if ( side == Side::Positive && sd->dist <= 0 )
        return false;
    if ( side == Side::Negative && sd->dist >= 0 )
        return false;
    return true;
}

VertBitSet findInnerShellVerts( const Mesh & mesh, const Mesh & shell, Side side )
{
    MR_TIMER
    VertBitSet res( shell.topology.vertSize() );
    BitSetParallelFor( shell.topology.getValidVerts(), [&]( VertId v )
    {
        if ( isInnerShellVert( mesh, shell.points[v], side ) )
            res.set( v );
    } );
    return res;
}

FaceBitSet findInnerShellFacesWithSplits( const Mesh & mesh, Mesh & shell, Side side )
{
    MR_TIMER
    const auto innerVerts = findInnerShellVerts( mesh, shell, side );

    // find all edges connecting inner and not-inner vertices
    UndirectedEdgeBitSet ues( shell.topology.undirectedEdgeSize() );
    BitSetParallelForAll( ues, [&]( UndirectedEdgeId ue )
    {
        if ( contains( innerVerts, shell.topology.org( ue ) ) != contains( innerVerts, shell.topology.dest( ue ) ) )
            ues.set( ue );
    } );
    //...
    return getIncidentFaces( shell.topology, innerVerts );
}

} //namespace MR
