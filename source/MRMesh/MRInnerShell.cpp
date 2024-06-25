#include "MRInnerShell.h"
#include "MRMesh.h"
#include "MRMeshProject.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

namespace MR
{

bool isInnerShellVert( const MeshPart & mp, const Vector3f & shellPoint, const FindInnerShellSettings & settings )
{
    const auto projRes = findProjection( shellPoint, mp, settings.maxDistSq );
    if ( !( projRes.distSq < settings.maxDistSq ) )
        return false;
    if ( projRes.mtp.isBd( mp.mesh.topology, mp.region ) )
        return false;

    const bool outside = settings.meshHasSelfIntersections ?
        mp.mesh.isOutside( shellPoint ) :
        mp.mesh.isOutsideByProjNorm( shellPoint, projRes, mp.region );
    if ( settings.side == Side::Positive && !outside )
        return false;
    if ( settings.side == Side::Negative && outside )
        return false;
    return true;
}

VertBitSet findInnerShellVerts( const MeshPart & mp, const Mesh & shell, const FindInnerShellSettings & settings )
{
    MR_TIMER
    VertBitSet res( shell.topology.vertSize() );
    BitSetParallelFor( shell.topology.getValidVerts(), [&]( VertId v )
    {
        if ( isInnerShellVert( mp, shell.points[v], settings ) )
            res.set( v );
    } );
    return res;
}

FaceBitSet findInnerShellFacesWithSplits( const MeshPart & mp, Mesh & shell, const FindInnerShellSettings & settings )
{
    MR_TIMER
    const auto innerVerts = findInnerShellVerts( mp, shell, settings );

    // find all edges connecting inner and not-inner vertices
    UndirectedEdgeBitSet ues( shell.topology.undirectedEdgeSize() );
    BitSetParallelForAll( ues, [&]( UndirectedEdgeId ue )
    {
        if ( contains( innerVerts, shell.topology.org( ue ) ) != contains( innerVerts, shell.topology.dest( ue ) ) )
            ues.set( ue );
    } );

    std::vector<EdgePoint> splitEdges;
    splitEdges.reserve( ues.count() );
    for ( EdgeId e : ues )
        splitEdges.emplace_back( e, 0 );

    // find split-point on each edge
    ParallelFor( splitEdges, [&]( size_t i )
    {
        EdgeId e = splitEdges[i].e;
        if ( !contains( innerVerts, shell.topology.org( e ) ) )
            e = e.sym();
        assert( contains( innerVerts, shell.topology.org( e ) ) );
        assert( !contains( innerVerts, shell.topology.dest( e ) ) );
        auto a = shell.orgPnt( e );
        auto b = shell.destPnt( e );
        float av = 0, bv = 1;
        // binary search
        for ( int j = 0; j < 8; ++j )
        {
            const auto v = 0.5f * ( av + bv );
            const auto p = ( 1 - v ) * a + v * b;
            if ( isInnerShellVert( mp, p, settings ) )
                av = v;
            else
                bv = v;
        }
        splitEdges[i] = EdgePoint( e, 0.5f * ( av + bv ) );
    } );

    // perform actual splitting
    for ( const auto & ep : splitEdges )
        shell.splitEdge( ep.e, shell.edgePoint( ep ) );

    return getIncidentFaces( shell.topology, innerVerts );
}

} //namespace MR
