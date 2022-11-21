#pragma once

#include "MRId.h"

namespace MR
{

/// encodes a point on an edge of mesh or of polyline
struct EdgePoint
{
    EdgeId e;
    float a = 0; ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )

    [[nodiscard]] EdgePoint() = default;
    [[nodiscard]] EdgePoint( EdgeId e, float a ) : e( e ), a( a ) { }
    [[nodiscard]] MRMESH_API EdgePoint( const MeshTopology & topology, VertId v );
    [[nodiscard]] MRMESH_API EdgePoint( const PolylineTopology & topology, VertId v );

    /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
    [[nodiscard]] MRMESH_API VertId inVertex( const MeshTopology & topology ) const;
    /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
    [[nodiscard]] MRMESH_API VertId inVertex( const PolylineTopology & topology ) const;
    /// returns one of two edge vertices, closest to this point
    [[nodiscard]] MRMESH_API VertId getClosestVertex( const MeshTopology & topology ) const;
    /// returns one of two edge vertices, closest to this point
    [[nodiscard]] MRMESH_API VertId getClosestVertex( const PolylineTopology & topology ) const;
    /// returns true if the point is in a vertex
    [[nodiscard]] MRMESH_API bool inVertex() const;

    /// represents the same point relative to sym edge in
    [[nodiscard]] EdgePoint sym() const { return EdgePoint{ e.sym(), 1 - a }; }
    bool operator==( const EdgePoint& rhs ) const { return ( (( e == rhs.e ) && ( a == rhs.a )) ||
        (( e == rhs.e.sym() ) && ( a == (1.f - rhs.a) ))); }
};

/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related EdgePoint
[[nodiscard]] MRMESH_API bool fromSameTriangle( const MeshTopology & topology, EdgePoint & a, EdgePoint & b );
/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related EdgePoint
[[nodiscard]] inline bool fromSameTriangle( const MeshTopology & topology, EdgePoint && a, EdgePoint && b ) { return fromSameTriangle( topology, a, b ); }

} // namespace MR
