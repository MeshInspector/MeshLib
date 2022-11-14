#pragma once

#include "MRId.h"

namespace MR
{

/// encodes a point on a mesh edge
/// \ingroup MeshGroup
struct MeshEdgePoint
{
    EdgeId e;
    float a = 0; ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )

    [[nodiscard]] MeshEdgePoint() = default;
    [[nodiscard]] MeshEdgePoint( EdgeId e, float a ) : e( e ), a( a ) { }
    [[nodiscard]] MRMESH_API MeshEdgePoint( const MeshTopology & topology, VertId v );

    /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
    [[nodiscard]] MRMESH_API VertId inVertex( const MeshTopology & topology ) const;
    /// returns one of two edge vertices, closest to this point
    [[nodiscard]] MRMESH_API VertId getClosestVertex( const MeshTopology & topology ) const;
    /// returns true if the point is in a vertex
    [[nodiscard]] MRMESH_API bool inVertex() const;

    /// represents the same point relative to sym edge in
    [[nodiscard]] MeshEdgePoint sym() const { return MeshEdgePoint{ e.sym(), 1 - a }; }
    bool operator==( const MeshEdgePoint& rhs ) const { return ( (( e == rhs.e ) && ( a == rhs.a )) ||
        (( e == rhs.e.sym() ) && ( a == (1.f - rhs.a) ))); }
};

/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related MeshEdgePoint
[[nodiscard]] MRMESH_API bool fromSameTriangle( const MeshTopology & topology, MeshEdgePoint & a, MeshEdgePoint & b );
/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related MeshEdgePoint
[[nodiscard]] inline bool fromSameTriangle( const MeshTopology & topology, MeshEdgePoint && a, MeshEdgePoint && b ) { return fromSameTriangle( topology, a, b ); }

} // namespace MR
