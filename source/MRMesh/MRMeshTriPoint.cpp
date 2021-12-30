#include "MRMeshTriPoint.h"
#include "MRMeshTopology.h"

namespace MR
{

VertId MeshTriPoint::inVertex( const MeshTopology & topology ) const
{
    switch( bary.inVertex() )
    {
    case 0:
        return topology.org( e );
    case 1:
        return topology.dest( e );
    case 2:
        return topology.dest( topology.next( e ) );
    }
    return {};
}

std::optional<MeshEdgePoint> MeshTriPoint::onEdge( const MeshTopology & topology ) const
{
    switch( bary.onEdge() )
    {
    case 0: // if ( a + b + eps >= 1 )
        return MeshEdgePoint{ topology.prev( e.sym() ), bary.b };
    case 1: // a <= eps
        return MeshEdgePoint{ topology.next( e ).sym(), 1 - bary.b };
    case 2: // b <= eps
        return MeshEdgePoint{ e, bary.a };
    }
    return {};
}

bool MeshTriPoint::isBd( const MeshTopology & topology, const FaceBitSet * region ) const
{
    if ( auto v = inVertex( topology ) )
        return topology.isBdVertex( v, region );
    if ( auto oe = onEdge( topology ) )
        return topology.isBdEdge( oe->e, region );
    return false;
}

MeshTriPoint MeshTriPoint::lnext( const MeshTopology & topology ) const
{
    MeshTriPoint res;
    res.e = topology.prev( e.sym() );
    res.bary.a = bary.b;
    res.bary.b = 1 - bary.a - bary.b;
    return res;
}

MeshTriPoint MeshTriPoint::canonical( const MeshTopology & topology ) const
{
    const auto e0 = topology.edgeWithLeft( topology.left( e ) );
    MeshTriPoint res = *this;
    for ( int i = 0; i < 2 && res.e != e0; ++i )
    {
        res = res.lnext( topology );
    }
    assert( res.e == e0 );
    return res;
}

std::optional<MeshTriPoint> getVertexAsMeshTriPoint( const MeshTopology & topology, EdgeId e, VertId v )
{
    VertId tv[3];
    topology.getLeftTriVerts( e, tv );
    if ( tv[0] == v )
        return MeshTriPoint( e, { 0, 0 } );
    if ( tv[1] == v )
        return MeshTriPoint( e, { 1, 0 } );
    if ( tv[2] == v )
        return MeshTriPoint( e, { 0, 1 } );
    return {};
}

static bool vertEdge2MeshTriPoints( const MeshTopology & topology, VertId av, const MeshEdgePoint & be, MeshTriPoint & a, MeshTriPoint & b )
{
    if ( topology.org( be.e ) == av )
    {
        a = MeshTriPoint( be.e, { 0, 0 } );
        b = MeshTriPoint( be );
        return true;
    }
    if ( topology.dest( be.e ) == av )
    {
        a = MeshTriPoint( be.e, { 1, 0 } );
        b = MeshTriPoint( be );
        return true;
    }
    if ( topology.left( be.e ) && topology.dest( topology.next( be.e ) ) == av )
    {
        a = MeshTriPoint( be.e, { 0, 1 } );
        b = MeshTriPoint( be );
        return true;
    }
    if ( topology.right( be.e ) && topology.dest( topology.prev( be.e ) ) == av )
    {
        a = MeshTriPoint( be.e.sym(), { 0, 1 } );
        b = MeshTriPoint( be.sym() );
        return true;
    }
    return false;
}

static bool edgePoint2MeshTriPoint( const MeshTopology & topology, const MeshEdgePoint & ae, FaceId f, MeshTriPoint & a )
{
    if ( topology.left( ae.e ) == f )
    {
        a = MeshTriPoint( ae );
        return true;
    }
    if ( topology.right( ae.e ) == f )
    {
        a = MeshTriPoint( ae.sym() );
        return true;
    }
    return false;
}

bool fromSameTriangle( const MeshTopology & topology, MeshTriPoint & a, MeshTriPoint & b )
{
    if ( auto av = a.inVertex( topology ) )
    {
        if ( auto bv = b.inVertex( topology ) )
        {
            // a in vertex, b in vertex
            if ( av == bv )
            {
                a = b = MeshTriPoint( topology.edgeWithOrg( av ), { 0, 0 } );
                return true;
            }
            if ( auto e = topology.findEdge( av, bv ) )
            {
                a = MeshTriPoint( e, { 0, 0 } );
                b = MeshTriPoint( e, { 1, 0 } );
                return true;
            }
            return false;
        }
        if ( auto be = b.onEdge( topology ) )
        {
            // a in vertex, b on edge
            return vertEdge2MeshTriPoints( topology, av, *be, a, b );
        }
        // a in vertex, b in triangle
        if ( auto mtp = getVertexAsMeshTriPoint( topology, b.e, av ) )
        {
            a = *mtp;
            return true;
        }
        return false;
    }
    if ( auto ae = a.onEdge( topology ) )
    {
        if ( auto bv = b.inVertex( topology ) )
        {
            // a on edge, b in vertex
            return vertEdge2MeshTriPoints( topology, bv, *ae, b, a );
        }
        if ( auto be = b.onEdge( topology ) )
        {
            // a on edge, b on edge
            const auto al = topology.left( ae->e );
            const auto ar = topology.right( ae->e );
            const auto bl = topology.left( be->e );
            const auto br = topology.right( be->e );
            if ( al && al == bl )
            {
                a = MeshTriPoint( *ae );
                b = MeshTriPoint( *be );
                return true;
            }
            if ( al && al == br )
            {
                a = MeshTriPoint( *ae );
                b = MeshTriPoint( be->sym() );
                return true;
            }
            if ( ar && ar == bl )
            {
                a = MeshTriPoint( ae->sym() );
                b = MeshTriPoint( *be );
                return true;
            }
            if ( ar && ar == br )
            {
                a = MeshTriPoint( ae->sym() );
                b = MeshTriPoint( be->sym() );
                return true;
            }
            return false;
        }
        // a on edge, b in triangle
        return edgePoint2MeshTriPoint( topology, *ae, topology.left( b.e ), a );
    }
    if ( auto bv = b.inVertex( topology ) )
    {
        // a in triangle, b in vertex
        if ( auto mtp = getVertexAsMeshTriPoint( topology, a.e, bv ) )
        {
            b = *mtp;
            return true;
        }
        return false;
    }
    if ( auto be = b.onEdge( topology ) )
    {
        // a in triangle, b on edge
        return edgePoint2MeshTriPoint( topology, *be, topology.left( a.e ), b );
    }
    // a in triangle, b in triangle
    return topology.left( a.e ) == topology.left( b.e );
}

} // namespace MR
