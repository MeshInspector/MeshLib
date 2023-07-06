#include "MRInnerShell.h"
#include "MRMesh.h"
#include "MRMeshProject.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
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

FaceBitSet findInnerShellFacesWithSplits( const Mesh & mesh, Mesh & shell, float offset )
{
    MR_TIMER
    const auto side = offset > 0 ? Side::Positive : Side::Negative;
    const auto innerVerts = findInnerShellVerts( mesh, shell, side );

    // find all edges connecting inner and not-inner vertices
    UndirectedEdgeBitSet ues( shell.topology.undirectedEdgeSize() );
    BitSetParallelForAll( ues, [&]( UndirectedEdgeId ue )
    {
        if ( contains( innerVerts, shell.topology.org( ue ) ) != contains( innerVerts, shell.topology.dest( ue ) ) )
            ues.set( ue );
    } );

    // for each edge to be split, stores the target point
    struct SplitEdge
    {
        EdgeId e;
        Vector3f p;
    };
    std::vector<SplitEdge> splitEdges;
    splitEdges.reserve( ues.count() );
    for ( EdgeId e : ues )
        splitEdges.push_back( { e } );

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
            if ( isInnerShellVert( mesh, p, side ) )
                av = v;
            else
                bv = v;
        }
        // shift found point to have exactly offset-distance
        auto pt = shell.edgePoint( EdgePoint( e, 0.5f * ( av + bv ) ) );
        auto prj = findProjection( pt, mesh );
        if ( prj.distSq > 0 )
            pt = pt + ( pt - prj.proj.point ).normalized() * std::abs( offset );
        splitEdges[i] = SplitEdge( e, pt );
    } );

    // perform actual splitting
    for ( const auto & se : splitEdges )
        shell.splitEdge( se.e, se.p );

    return getIncidentFaces( shell.topology, innerVerts );
}

} //namespace MR
