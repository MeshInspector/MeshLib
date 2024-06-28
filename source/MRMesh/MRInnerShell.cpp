#include "MRInnerShell.h"
#include "MRMesh.h"
#include "MRMeshProject.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRTimer.h"

namespace MR
{

ShellVertexInfo classifyShellVert( const MeshPart & mp, const Vector3f & shellPoint, const FindInnerShellSettings & settings )
{
    ShellVertexInfo res;

    MeshProjectionResult projRes;
    if ( !settings.useWindingNumber || settings.maxDistSq < FLT_MAX )
    {
        projRes = findProjection( shellPoint, mp, settings.maxDistSq );
        if ( !( projRes.distSq < settings.maxDistSq ) )
            return res;
    }
    res.inRange = true;

    if ( settings.useWindingNumber )
    {
        const bool outside = mp.mesh.isOutside( shellPoint, settings.windingNumberThreshold );
        res.rightSide = outside == ( settings.side == Side::Positive );
        return res;
    }

    res.projOnBd = projRes.mtp.isBd( mp.mesh.topology, mp.region );

    const bool outside = mp.mesh.isOutsideByProjNorm( shellPoint, projRes, mp.region );
    res.rightSide = outside == ( settings.side == Side::Positive );
    return res;
}

VertBitSet findInnerShellVerts( const MeshPart & mp, const Mesh & shell, const FindInnerShellSettings & settings )
{
    MR_TIMER
    VertBitSet mySide( shell.topology.vertSize() ), notBd( shell.topology.vertSize() );
    BitSetParallelFor( shell.topology.getValidVerts(), [&]( VertId v )
    {
        const auto info = classifyShellVert( mp, shell.points[v], settings );
        if ( info.inRange && !info.projOnBd )
        {
            notBd.set( v );
            if ( info.rightSide )
                mySide.set( v );
        }
    } );

    const auto largeComps = MeshComponents::getLargeComponentVerts( shell, settings.minVertsInComp, &notBd );
    mySide &= largeComps;
    const auto largeMySide = MeshComponents::getLargeComponentVerts( shell, settings.minVertsInComp, &mySide );
    const auto otherSide = largeComps - mySide;
    const auto largeOtherSide = MeshComponents::getLargeComponentVerts( shell, settings.minVertsInComp, &otherSide );

    auto res = largeMySide | ( otherSide - largeOtherSide );
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
            if ( classifyShellVert( mp, p, settings ).valid() )
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
