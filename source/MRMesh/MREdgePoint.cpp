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

template <typename T>
bool same( const MeshTopology & topology, const EdgePointT<T>& lhs, const EdgePointT<T>& rhs )
{
    if ( !lhs )
        return !rhs;
    if ( auto v = lhs.inVertex( topology ) )
        return v == rhs.inVertex( topology );

    return lhs == rhs || lhs == rhs.sym();
}

template <typename T>
static bool vertEdge2MeshEdgePoints( const MeshTopology & topology, VertId av, EdgePointT<T> & a, EdgePointT<T> & b )
{
    if ( topology.org( b.e ) == av )
    {
        a = EdgePointT<T>( b.e, 0 );
        return true;
    }
    if ( topology.dest( b.e ) == av )
    {
        a = EdgePointT<T>( b.e, 1 );
        return true;
    }
    if ( topology.left( b.e ) && topology.dest( topology.next( b.e ) ) == av )
    {
        a = EdgePointT<T>( topology.next( b.e ).sym(), 0 );
        return true;
    }
    if ( topology.right( b.e ) && topology.dest( topology.prev( b.e ) ) == av )
    {
        a = EdgePointT<T>( topology.prev( b.e ).sym(), 0 );
        b = b.sym();
        return true;
    }
    return false;
}

template <typename T>
bool fromSameTriangle( const MeshTopology & topology, EdgePointT<T> & a, EdgePointT<T> & b )
{
    if ( auto av = a.inVertex( topology ) )
    {
        if ( auto bv = b.inVertex( topology ) )
        {
            // a in vertex, b in vertex
            if ( av == bv )
            {
                a = b = EdgePointT<T>( topology.edgeWithOrg( av ), 0 );
                return true;
            }
            if ( auto e = topology.findEdge( av, bv ) )
            {
                a = EdgePointT<T>( e, 0 );
                b = EdgePointT<T>( e, 1 );
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

template struct EdgePointT<float>;
template struct EdgePointT<double>;

template MRMESH_API bool same( const MeshTopology & topology, const EdgePointT<float>& lhs, const EdgePointT<float>& rhs );
template MRMESH_API bool same( const MeshTopology & topology, const EdgePointT<double>& lhs, const EdgePointT<double>& rhs );
template MRMESH_API bool fromSameTriangle( const MeshTopology & topology, EdgePointT<float> & a, EdgePointT<float> & b );
template MRMESH_API bool fromSameTriangle( const MeshTopology & topology, EdgePointT<double> & a, EdgePointT<double> & b );

} // namespace MR
