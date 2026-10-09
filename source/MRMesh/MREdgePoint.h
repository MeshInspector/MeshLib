#pragma once

#include "MRId.h"
#include "MRSegmPoint.h"

namespace MR
{

/// encodes a point on an edge of mesh or of polyline
template <typename T>
struct EdgePointT
{
    EdgeId e;
    SegmPoint<T> a; ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )

    [[nodiscard]] EdgePointT() = default;
    [[nodiscard]] EdgePointT( EdgeId e, T a ) : e( e ), a( a ) { }
    [[nodiscard]] MRMESH_API EdgePointT( const MeshTopology & topology, VertId v );
    [[nodiscard]] MRMESH_API EdgePointT( const PolylineTopology & topology, VertId v );

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
    [[nodiscard]] EdgePointT sym() const { return EdgePointT{ e.sym(), 1 - a }; }
    /// returns true if two edge-points are equal including equal not-unique representation
    [[nodiscard]] bool operator==( const EdgePointT& rhs ) const = default;
};

/// returns true if two edge-points are equal considering different representations
template <typename T>
[[nodiscard]] MRMESH_API bool same( const MeshTopology & topology, const EdgePointT<T>& lhs, const EdgePointT<T>& rhs );
MR_BIND_TEMPLATE( bool same( const MeshTopology & topology, const EdgePointT<float>& lhs, const EdgePointT<float>& rhs ) )
MR_BIND_TEMPLATE( bool same( const MeshTopology & topology, const EdgePointT<double>& lhs, const EdgePointT<double>& rhs ) )

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
template <typename T>
struct EdgeSegmentT
{
    /// id of the edge
    EdgeId e;
    /// start of the segment
    SegmPoint<T> a{ 0 };
    /// end of the segment
    SegmPoint<T> b{ 1 };
    [[nodiscard]] EdgeSegmentT() = default;
    [[nodiscard]] EdgeSegmentT( EdgeId e, T a = 0, T b = 1 ) : e( e ), a( a ), b( b ) { assert( valid() ); };
    /// returns starting EdgePoint
    [[nodiscard]] EdgePointT<T> edgePointA() const { return { e, a }; }
    /// returns ending EdgePoint
    [[nodiscard]] EdgePointT<T> edgePointB() const { return { e, b }; }
    /// returns true if the edge is valid and start point is less than end point
    [[nodiscard]] bool valid() const { return e.valid() && a <= b; }

    bool operator==( const EdgeSegmentT& rhs ) const = default;
    /// represents the same segment relative to sym edge in
    [[nodiscard]] EdgeSegmentT sym() const { return EdgeSegmentT{ e.sym(), b.sym(), a.sym() }; }
};

/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related EdgePoint
template <typename T>
[[nodiscard]] MRMESH_API bool fromSameTriangle( const MeshTopology & topology, EdgePointT<T> & a, EdgePointT<T> & b );
MR_BIND_TEMPLATE( bool fromSameTriangle( const MeshTopology & topology, EdgePointT<float> & a, EdgePointT<float> & b ) )
MR_BIND_TEMPLATE( bool fromSameTriangle( const MeshTopology & topology, EdgePointT<double> & a, EdgePointT<double> & b ) )
/// returns true if points a and b are located on a boundary of the same triangle;
/// \details if true a.e and b.e are updated to have that triangle on the left
/// \related EdgePoint
template <typename T>
[[nodiscard]] inline bool fromSameTriangle( const MeshTopology & topology, EdgePointT<T> && a, EdgePointT<T> && b ) { return fromSameTriangle( topology, a, b ); }
MR_BIND_TEMPLATE( bool fromSameTriangle( const MeshTopology & topology, EdgePointT<float> && a, EdgePointT<float> && b ) )
MR_BIND_TEMPLATE( bool fromSameTriangle( const MeshTopology & topology, EdgePointT<double> && a, EdgePointT<double> && b ) )

} // namespace MR
