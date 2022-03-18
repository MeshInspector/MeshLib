#pragma once
#include "MRFaceFace.h"
#include "MRPolyline2.h"

namespace MR
{

// finds all pairs of colliding edges from two 2d polylines
MRMESH_API std::vector<UndirectedEdgeUndirectedEdge> findCollidingEdges( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr, // rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
    bool firstIntersectionOnly = false );  // if true then the function returns at most one pair of intersecting edges and returns faster
// the same, but returns one bite set per polyline with colliding edges
MRMESH_API std::pair<UndirectedEdgeBitSet, UndirectedEdgeBitSet> findCollidingEdgesBitsets( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr );

// finds all pairs of colliding edges from 2d polyline
MRMESH_API std::vector<UndirectedEdgeUndirectedEdge> findSelfCollidingEdges( const Polyline2& mp );
// the same but returns the union of all self-intersecting edges
MRMESH_API UndirectedEdgeBitSet findSelfCollidingEdgesBS( const Polyline2& mp );

// checks that arbitrary 2d polyline A is inside of closed 2d polyline B
MRMESH_API bool isInside( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr ); // rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation

} //namespace MR