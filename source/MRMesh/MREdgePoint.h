#pragma once

#include "MRId.h"
#include "MRSegmPoint.h"

namespace MR
{

/// encodes a point on an edge of mesh or of polyline
struct EdgePoint
{
    EdgeId e;
    SegmPointf a; ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )

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
    [[nodiscard]] bool inVertex() const { return a.inVertex() >= 0; }
    /// sets this to the closest end of the edge
    MRMESH_API void moveToClosestVertex();
    /// returns true if the point is on the boundary of the region (or for whole mesh if region is nullptr)
    [[nodiscard]] MRMESH_API bool isBd( const MeshTopology & topology, const FaceBitSet * region = nullptr ) const;

    /// consider this valid if the edge ID is valid
    [[nodiscard]] bool valid() const { return e.valid(); }
    [[nodiscard]] explicit operator bool() const { return e.valid(); }

    /// represents the same point relative to sym edge in
    [[nodiscard]] EdgePoint sym() const { return EdgePoint{ e.sym(), 1 - a }; }
    /// returns true if two edge-points are equal including equal not-unique representation
    [[nodiscard]] bool operator==( const EdgePoint& rhs ) const = default;
};

/// returns true if two edge-points are equal considering different representations
[[nodiscard]] MRMESH_API bool same( const MeshTopology & topology, const EdgePoint& lhs, const EdgePoint& rhs );

/// two edge-points (e.g. representing collision point of two edges)
struct EdgePointPair
{
    EdgePoint a;
    EdgePoint b;
    EdgePointPair() = default;
    EdgePointPair( EdgePoint ia, EdgePoint ib ) : a( ia ), b( ib ) {}
    /// returns true if two edge-point pairs are equal including equal not-unique representation
    bool operator==( const EdgePointPair& rhs ) const = default;
};

/// Represents a segment on one edge
struct EdgeSegment
{
    // id of the edge
    EdgeId e;
    // start of the segment
    SegmPointf a{ 0.0f };
    // end of the segment
    SegmPointf b{ 1.0f };
    [[nodiscard]] EdgeSegment() = default;
    [[nodiscard]] EdgeSegment( EdgeId e, float a = 0.0f, float b = 1.0f ) : e( e ), a( a ), b( b ) { assert( valid() ); };
    // returns starting EdgePoint
    [[nodiscard]] EdgePoint edgePointA() const { return { e, a }; }
    // returns ending EdgePoint
    [[nodiscard]] EdgePoint edgePointB() const { return { e, b }; }
    // returns true if the edge is valid and start point is less than end point
    [[nodiscard]] bool valid() const { return e.valid() && a <= b; }
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
