#pragma once

#include "MRId.h"
#include "MRTriPoint.h"
#include "MRMeshEdgePoint.h"

#include <optional>

namespace MR
{

/// encodes a point inside a triangular mesh face using barycentric coordinates
/// \ingroup MeshGroup
/// \details Notations used below: \n
///   v0 - the value in org( e ) \n
///   v1 - the value in dest( e ) \n
///   v2 - the value in dest( next( e ) )
struct MeshTriPoint
{
    EdgeId e; ///< left face of this edge is considered
    /// barycentric coordinates
    /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
    /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
    /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
    TriPointf bary;

    MeshTriPoint() = default;
    MeshTriPoint( EdgeId e, TriPointf bary ) : e( e ), bary( bary ) { }
    MeshTriPoint( const MeshEdgePoint & ep ) : e( ep.e ), bary( ep.a, 0 ) { }

    /// given a point coordinates computes its barycentric coordinates
    template< typename T >
    MeshTriPoint( EdgeId e, const Vector3<T> & p, const Vector3<T> & v0, const Vector3<T> & v1, const Vector3<T> & v2 ) : e( e ), bary( p, v0, v1, v2 ) { }

    /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
    [[nodiscard]] MRMESH_API VertId inVertex( const MeshTopology & topology ) const;
    /// returns valid value if the point is on edge, otherwise returns null optional
    [[nodiscard]] MRMESH_API std::optional<MeshEdgePoint> onEdge( const MeshTopology & topology ) const;
    /// returns true if the point is in vertex on on edge, and that location is on the boundary of the region
    [[nodiscard]] MRMESH_API bool isBd( const MeshTopology & topology, const FaceBitSet * region = nullptr ) const;

    /// represents the same point relative to next edge in the same triangle
    [[nodiscard]] MRMESH_API MeshTriPoint lnext( const MeshTopology & topology ) const;
    /// represents the same point relative to the topology.edgeWithLeft( topology.left( e ) )
    [[nodiscard]] MRMESH_API MeshTriPoint canonical( const MeshTopology & topology ) const;
};

/// \related MeshTriPoint
/// \{

/// returns true if points a and b are located insides or on a boundary of the same triangle;
/// if true a.e and b.e are updated to have that triangle on the left
[[nodiscard]] MRMESH_API bool fromSameTriangle( const MeshTopology & topology, MeshTriPoint & a, MeshTriPoint & b );

/// returns MeshTriPoint representation of given vertex with given edge field; or null if it is not possible
[[nodiscard]] MRMESH_API std::optional<MeshTriPoint> getVertexAsMeshTriPoint( const MeshTopology & topology, EdgeId e, VertId v );

/// \}

} // namespace MR
