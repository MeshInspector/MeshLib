#pragma once

#include "MRId.h"

namespace MR
{

// encodes a point on a mesh edge
struct MeshEdgePoint
{
    EdgeId e;
    float a = 0; // a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )

    MeshEdgePoint() = default;
    MeshEdgePoint( EdgeId e, float a ) : e( e ), a( a ) { }

    // returns valid vertex id if the point is in vertex, otherwise returns invalid id
    [[nodiscard]] MRMESH_API VertId inVertex( const MeshTopology & topology ) const;
    // returns one of two edge vertices, closest to this point
    [[nodiscard]] MRMESH_API VertId getClosestVertex( const MeshTopology & topology ) const;
    // just returns true of false
    [[nodiscard]] MRMESH_API bool inVertex() const;

    // represents the same point relative to sym edge in
    [[nodiscard]] MeshEdgePoint sym() const { return MeshEdgePoint{ e.sym(), 1 - a }; }
    bool operator==( const MeshEdgePoint& rhs ) const { return ( (( e == rhs.e ) && ( a == rhs.a )) ||
        (( e == rhs.e.sym() ) && ( a == (1.f - rhs.a) ))); }
};

// returns true if points a and b are located on a boundary of the same triangle;
// if true a.e and b.e are updated to have that triangle on the left
[[nodiscard]] MRMESH_API bool fromSameTriangle( const MeshTopology & topology, MeshEdgePoint & a, MeshEdgePoint & b );

} // namespace MR
