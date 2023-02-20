#include "MRMeshTrimWithPlane.h"
#include "MRMesh.h"
#include "MRPlane3.h"
#include "MRVector2.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old, float eps, Vector<UVCoord, VertId>* uvCoords )
{
    MR_TIMER
    assert( eps >= 0 );
    if ( uvCoords && uvCoords->size() < mesh.points.size() )
        uvCoords = nullptr;

    VertBitSet positiveVerts( mesh.topology.vertSize() );
    VertBitSet negativeVerts( positiveVerts.size() );

    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
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
        VertId vo = mesh.topology.org( e );
        VertId vd = mesh.topology.dest( e );
        auto po = mesh.points[vo];
        auto pd = mesh.points[vd];
        auto o = plane.distance( po );
        auto d = plane.distance( pd );
        if ( o * d >= 0 )
            continue; // this may happen if origin or destination were projected on plane before
        if ( d > 0 )
        {
            e = e.sym();
            std::swap( vo, vd );
            std::swap( po, pd );
            std::swap(  o,  d );
        }
        assert( o > 0 && d < 0 );
        if ( o <= eps )
        {
            // project existing positive vertex on plane if it was close
            mesh.points[vo] = plane.project( mesh.points[vo] );
        }
        else if ( d >= -eps )
        {
            // project existing negative vertex on plane if it was close
            mesh.points[vd] = plane.project( mesh.points[vd] );
            negativeVerts.reset( vd );
            e = e.sym();
        }
        else
        {
            // introduce new vertex if both existing vertices are far from plane
            const auto p = ( o * pd - d * po ) / ( o - d );
            mesh.splitEdge( e, p, nullptr, new2Old );

            if ( uvCoords )
            {
                const auto uvCoord = ( o * ( *uvCoords )[vd] - d * ( *uvCoords )[vo] ) / ( o - d );
                uvCoords->push_back( uvCoord );
            }
        }
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

void trimWithPlane( Mesh& mesh, const Plane3f & plane, UndirectedEdgeBitSet * outCutEdges, FaceHashMap * new2Old, float eps, Vector<UVCoord, VertId>* uvCoords )
{
    MR_TIMER
    const auto posFaces = subdivideWithPlane( mesh, plane, new2Old, eps, uvCoords );
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

void trimWithPlane( Mesh& mesh, const Plane3f & plane, std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old, float eps, Vector<UVCoord, VertId>* uvCoords )
{
    MR_TIMER
    const auto posFaces = subdivideWithPlane( mesh, plane, new2Old, eps, uvCoords );
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
