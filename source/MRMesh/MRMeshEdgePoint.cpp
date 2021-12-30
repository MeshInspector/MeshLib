#include "MRMeshEdgePoint.h"
#include "MRMeshTopology.h"

namespace MR
{

static constexpr auto eps = 10 * std::numeric_limits<float>::epsilon();

VertId MeshEdgePoint::inVertex( const MeshTopology & topology ) const
{
    if ( a <= eps )
        return topology.org( e );
    if ( a + eps >= 1 )
        return topology.dest( e );
    return {};
}

bool MeshEdgePoint::inVertex() const
{
    return a <= eps || a + eps >= 1;
}

static bool vertEdge2MeshEdgePoints( const MeshTopology & topology, VertId av, MeshEdgePoint & a, MeshEdgePoint & b )
{
    if ( topology.org( b.e ) == av )
    {
        a = MeshEdgePoint( b.e, 0 );
        return true;
    }
    if ( topology.dest( b.e ) == av )
    {
        a = MeshEdgePoint( b.e, 1 );
        return true;
    }
    if ( topology.left( b.e ) && topology.dest( topology.next( b.e ) ) == av )
    {
        a = MeshEdgePoint( topology.next( b.e ).sym(), 0 );
        return true;
    }
    if ( topology.right( b.e ) && topology.dest( topology.prev( b.e ) ) == av )
    {
        a = MeshEdgePoint( topology.prev( b.e ).sym(), 0 );
        b = b.sym();
        return true;
    }
    return false;
}

bool fromSameTriangle( const MeshTopology & topology, MeshEdgePoint & a, MeshEdgePoint & b )
{
    if ( auto av = a.inVertex( topology ) )
    {
        if ( auto bv = b.inVertex( topology ) )
        {
            // a in vertex, b in vertex
            if ( av == bv )
            {
                a = b = MeshEdgePoint( topology.edgeWithOrg( av ), 0 );
                return true;
            }
            if ( auto e = topology.findEdge( av, bv ) )
            {
                a = MeshEdgePoint( e, 0 );
                b = MeshEdgePoint( e, 1 );
                return true;
            }
            return false;
        }
        // a in vertex, b on edge
        return vertEdge2MeshEdgePoints( topology, av, a, b );
    }
    if ( auto bv = b.inVertex( topology ) )
    {
        // a on edge, b in vertex
        return vertEdge2MeshEdgePoints( topology, bv, b, a );
    }
    // a on edge, b on edge
    const auto al = topology.left( a.e );
    const auto ar = topology.right( a.e );
    const auto bl = topology.left( b.e );
    const auto br = topology.right( b.e );
    if ( al && al == bl )
    {
        return true;
    }
    if ( al && al == br )
    {
        b = b.sym();
        return true;
    }
    if ( ar && ar == bl )
    {
        a = a.sym();
        return true;
    }
    if ( ar && ar == br )
    {
        a = a.sym();
        b = b.sym();
        return true;
    }
    return false;
}

} // namespace MR
