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

FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old, float eps, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
{
    MR_TIMER
    assert( eps >= 0 );

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
    const VertId firstNewVert( mesh.topology.vertSize() );
    auto isNewVert = [firstNewVert]( VertId v )
        { return v >= firstNewVert; };
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
            auto p = ( o * pd - d * po ) / ( o - d );
            auto dotCheck = dot( p - po, pd - po );
            if ( dotCheck <= 0 )
                p = po;
            else if ( dotCheck >= ( pd - po ).lengthSq() )
                p = pd;

            auto eNew = mesh.splitEdge( e, p, nullptr, new2Old );
            assert( isNewVert( mesh.topology.org( e ) ) );

            // make triangulation independent on the order of edge splitting
            const auto eNext = mesh.topology.next( e );
            if ( mesh.topology.left( e ) && isNewVert( mesh.topology.dest( eNext ) ) )
            {
                const auto ee = mesh.topology.next( eNext.sym() );
                if ( edgesToCut.test( ee.undirected() ) )
                    mesh.topology.flipEdge( mesh.topology.prev( eNext.sym() ) );
                else
                {
                    assert( edgesToCut.test( mesh.topology.next( ee ).undirected() ) );
                    mesh.topology.flipEdge( ee );
                }
            }

            if ( onEdgeSplitCallback )
                onEdgeSplitCallback( e, eNew, o / ( o - d ) );
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

void trimWithPlane( Mesh& mesh, const Plane3f & plane, UndirectedEdgeBitSet * outCutEdges, FaceHashMap * new2Old, float eps, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
{
    return trimWithPlane( mesh, TrimWithPlaneParams{ .plane = plane, .eps = eps, .onEdgeSplitCallback = onEdgeSplitCallback }, TrimOptionalOutput{ .outCutEdges = outCutEdges, .new2Old = new2Old } );
}

void trimWithPlane( Mesh& mesh, const Plane3f & plane, std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old, float eps, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
{
    return trimWithPlane( mesh, TrimWithPlaneParams{ .plane = plane, .eps = eps, .onEdgeSplitCallback = onEdgeSplitCallback }, TrimOptionalOutput{ .outCutContours = outCutContours, .new2Old = new2Old } );
}

void trimWithPlane( Mesh& mesh, const TrimWithPlaneParams& params, const TrimOptionalOutput& optOut )
{
    MR_TIMER
    MR_WRITER( mesh );
    const auto posFaces = subdivideWithPlane( mesh, params.plane, optOut.new2Old, params.eps, params.onEdgeSplitCallback );
    if ( optOut.outCutEdges )
        *optOut.outCutEdges = findRegionBoundaryUndirectedEdgesInsideMesh( mesh.topology, posFaces );

    if ( optOut.outCutContours )
    {
        *optOut.outCutContours = findLeftBoundaryInsideMesh( mesh.topology, posFaces );
#ifndef NDEBUG
        for ( const auto& c : *optOut.outCutContours )
            for ( [[maybe_unused]] EdgeId e : c )
            {
                assert( contains( posFaces, mesh.topology.left( e ) ) );
                assert( mesh.topology.right( e ) );
                assert( !contains( posFaces, mesh.topology.right( e ) ) );
            }
#endif
    }
    
    const auto otherFaces = mesh.topology.getValidFaces() - posFaces;
    if ( optOut.otherPart )
    {
        *optOut.otherPart = mesh;        
        if ( optOut.otherOutCutContours )
        {
            *optOut.otherOutCutContours = findLeftBoundaryInsideMesh( optOut.otherPart->topology, otherFaces );
#ifndef NDEBUG
            for ( const auto& c : *optOut.otherOutCutContours )
                for ( [[maybe_unused]] EdgeId e : c )
                {
                    assert( contains( otherFaces, optOut.otherPart->topology.left( e ) ) );
                    assert( optOut.otherPart->topology.right( e ) );
                    assert( !contains( otherFaces, optOut.otherPart->topology.right( e ) ) );
                }
#endif
        }
        optOut.otherPart->topology.deleteFaces( posFaces );
    }
    mesh.topology.deleteFaces( otherFaces );
#ifndef NDEBUG
    if ( optOut.outCutContours )
    {
        for ( const auto& c : *optOut.outCutContours )
            for ( [[maybe_unused]] EdgeId e : c )
            {
                assert( mesh.topology.left( e ) );
                assert( !mesh.topology.right( e ) );
            }
    }

    if ( optOut.otherPart && optOut.otherOutCutContours )
    {
        for ( const auto& c : *optOut.otherOutCutContours )
            for ( [[maybe_unused]] EdgeId e : c )
            {
                assert( optOut.otherPart->topology.left( e ) );
                assert( !optOut.otherPart->topology.right( e ) );
            }
    }

    if ( optOut.outCutEdges )
    {
        for ( [[maybe_unused]] EdgeId e : *optOut.outCutEdges )
            assert( mesh.topology.left( e ).valid() != mesh.topology.right( e ).valid() );
    }
#endif
    if ( optOut.new2Old )
    {
        for ( auto it = optOut.new2Old->begin(); it != optOut.new2Old->end(); )
            if ( mesh.topology.hasFace( it->first ) )
                ++it;
            else
            {
                if ( optOut.otherNew2Old )
                    optOut.otherNew2Old->insert_or_assign( it->first, it->second );

                it = optOut.new2Old->erase( it );
            }
    }
}

} //namespace MR
