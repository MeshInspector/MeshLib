#include "MREdgePoint.h"
#include "MRMeshTopology.h"
#include "MRPolylineTopology.h"

namespace MR
{

template <typename T>
EdgePointT<T>::EdgePointT( const MeshTopology & topology, VertId v ) : e( topology.edgeWithOrg( v ) )
{
}

template <typename T>
EdgePointT<T>::EdgePointT( const PolylineTopology & topology, VertId v ) : e( topology.edgeWithOrg( v ) )
{
}

template <typename T>
VertId EdgePointT<T>::inVertex( const MeshTopology & topology ) const
{
    switch ( a.inVertex() )
    {
    case 0:
        return topology.org( e );
    case 1:
        return topology.dest( e );
    default:
        return {};
    }
}

template <typename T>
VertId EdgePointT<T>::inVertex( const PolylineTopology & topology ) const
{
    switch ( a.inVertex() )
    {
    case 0:
        return topology.org( e );
    case 1:
        return topology.dest( e );
    default:
        return {};
    }
}

template <typename T>
VertId EdgePointT<T>::getClosestVertex( const MeshTopology & topology ) const
{
    if ( 2 * a <= 1 )
        return topology.org( e );
    else
        return topology.dest( e );
}

template <typename T>
VertId EdgePointT<T>::getClosestVertex( const PolylineTopology & topology ) const
{
    if ( 2 * a <= 1 )
        return topology.org( e );
    else
        return topology.dest( e );
}

template <typename T>
void EdgePointT<T>::moveToClosestVertex()
{
    if ( 2 * a <= 1 )
        a = 0;
    else
        a = 1;
}

template <typename T>
bool EdgePointT<T>::isBd( const MeshTopology & topology, const FaceBitSet * region ) const
{
    if ( auto v = inVertex( topology ) )
        return topology.isBdVertex( v, region );
    return topology.isBdEdge( e, region );
}

template struct EdgePointT<float>;
template struct EdgePointT<double>;

bool same( const MeshTopology & topology, const EdgePoint& lhs, const EdgePoint& rhs )
{
    if ( !lhs )
        return !rhs;
    if ( auto v = lhs.inVertex( topology ) )
        return v == rhs.inVertex( topology );

    return lhs == rhs || lhs == rhs.sym();
}

static bool vertEdge2MeshEdgePoints( const MeshTopology & topology, VertId av, EdgePoint & a, EdgePoint & b )
{
    if ( topology.org( b.e ) == av )
    {
        a = EdgePoint( b.e, 0 );
        return true;
    }
    if ( topology.dest( b.e ) == av )
    {
        a = EdgePoint( b.e, 1 );
        return true;
    }
    if ( topology.left( b.e ) && topology.dest( topology.next( b.e ) ) == av )
    {
        a = EdgePoint( topology.next( b.e ).sym(), 0 );
        return true;
    }
    if ( topology.right( b.e ) && topology.dest( topology.prev( b.e ) ) == av )
    {
        a = EdgePoint( topology.prev( b.e ).sym(), 0 );
        b = b.sym();
        return true;
    }
    return false;
}

bool fromSameTriangle( const MeshTopology & topology, EdgePoint & a, EdgePoint & b )
{
    if ( auto av = a.inVertex( topology ) )
    {
        if ( auto bv = b.inVertex( topology ) )
        {
            // a in vertex, b in vertex
            if ( av == bv )
            {
                a = b = EdgePoint( topology.edgeWithOrg( av ), 0 );
                return true;
            }
            if ( auto e = topology.findEdge( av, bv ) )
            {
                a = EdgePoint( e, 0 );
                b = EdgePoint( e, 1 );
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
