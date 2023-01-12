#include "MRMeshTrimWithPlane.h"
#include "MRMesh.h"
#include "MRPlane3.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old )
{
    MR_TIMER
    VertBitSet positiveVerts( mesh.topology.lastValidVert() + 1 );
    VertBitSet negativeVerts( positiveVerts.size() );

    BitSetParallelForAll( positiveVerts, [&]( VertId v )
    {
        const auto d = plane.distance( mesh.points[v] );
        if ( d > 0 )
            positiveVerts.set( v );
        else if ( d < 0 )
            negativeVerts.set( v );
    } );

    UndirectedEdgeBitSet edgesToCut( mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( edgesToCut, [&]( UndirectedEdgeId ue )
    {
        const VertId o = mesh.topology.org( ue );
        if ( !o )
            return;
        const VertId d = mesh.topology.dest( ue );
        if ( !d )
            return;
        if ( ( positiveVerts.test( o ) && negativeVerts.test( d ) )
          || ( positiveVerts.test( d ) && negativeVerts.test( o ) ) )
            edgesToCut.set( ue );
    } );

    FaceBitSet positiveFaces( mesh.topology.lastValidFace() + 1 );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        VertId vs[3];
        mesh.topology.getTriVerts( f, vs );
        bool pos = false;
        bool neg = false;
        for ( VertId v : vs )
        {
            if ( positiveVerts.test( v ) )
                pos = true;
            else if ( negativeVerts.test( v ) )
                neg = true;
        }
        if ( pos && !neg )
            positiveFaces.set( f );
    } );

    MR_WRITER( mesh );
    for ( EdgeId e : edgesToCut )
    {
        const VertId vo = mesh.topology.org( e );
        const VertId vd = mesh.topology.dest( e );
        const auto po = mesh.points[vo];
        const auto pd = mesh.points[vd];
        const auto o = plane.distance( po );
        const auto d = plane.distance( pd );
        assert( o * d < 0 );
        const auto p = ( o * pd - d * po ) / ( o - d );
        mesh.splitEdge( e, p, nullptr, new2Old );
        for ( EdgeId ei : orgRing( mesh.topology, e ) )
        {
            const auto l = mesh.topology.left( ei );
            if ( l && !negativeVerts.test( mesh.topology.dest( ei ) )
                   && !negativeVerts.test( mesh.topology.dest( mesh.topology.next( ei ) ) ) )
                positiveFaces.autoResizeSet( l );
        }
    }

    return positiveFaces;
}

void trimWithPlane( Mesh& mesh, const Plane3f & plane, UndirectedEdgeBitSet * outCutEdges, FaceHashMap * new2Old )
{
    MR_TIMER
    const auto posFaces = subdivideWithPlane( mesh, plane, new2Old );
    if ( outCutEdges )
        *outCutEdges = findRegionBoundaryUndirectedEdgesInsideMesh( mesh.topology, posFaces );
    mesh.topology.deleteFaces( mesh.topology.getValidFaces() - posFaces );
#ifndef NDEBUG
    if ( outCutEdges )
    {
        for ( [[maybe_unused]] EdgeId e : *outCutEdges )
            assert( mesh.topology.left( e ).valid() !=mesh.topology.right( e ).valid() );
    }
#endif
    if ( new2Old )
    {
        for ( auto it = new2Old->begin(); it != new2Old->end(); )
            if ( mesh.topology.hasFace( it->first ) )
                ++it;
            else
                it = new2Old->erase( it );
    }
}

void trimWithPlane( Mesh& mesh, const Plane3f & plane, std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old )
{
    MR_TIMER
    const auto posFaces = subdivideWithPlane( mesh, plane, new2Old );
    if ( outCutContours )
    {
        *outCutContours = findRegionBoundaryInsideMesh( mesh.topology, posFaces );
#ifndef NDEBUG
        for ( const auto & c : *outCutContours )
            for ( [[maybe_unused]] EdgeId e : c )
            {
                assert( contains( posFaces, mesh.topology.left( e ) ) );
                assert( mesh.topology.right( e ) );
                assert( !contains( posFaces, mesh.topology.right( e ) ) );
            }
#endif
    }
    mesh.topology.deleteFaces( mesh.topology.getValidFaces() - posFaces );
#ifndef NDEBUG
    if ( outCutContours )
    {
        for ( const auto & c : *outCutContours )
            for ( [[maybe_unused]] EdgeId e : c )
            {
                assert( mesh.topology.left( e ) );
                assert( !mesh.topology.right( e ) );
            }
    }
#endif
    if ( new2Old )
    {
        for ( auto it = new2Old->begin(); it != new2Old->end(); )
            if ( mesh.topology.hasFace( it->first ) )
                ++it;
            else
                it = new2Old->erase( it );
    }
}

} //namespace MR
