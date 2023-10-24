#pragma once
#include "MRFaceFace.h"
#include "MRPolyline.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/**
 * \brief finds all pairs of colliding edges from two 2d polylines
 * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
 * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting edges and returns faster
 */
MRMESH_API std::vector<EdgePointPair> findCollidingEdgePairs( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr, bool firstIntersectionOnly = false );

/**
 * \brief finds all pairs of colliding edges from two 2d polylines
 * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
 * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting edges and returns faster
 */
MRMESH_API std::vector<UndirectedEdgeUndirectedEdge> findCollidingEdges( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr, bool firstIntersectionOnly = false );

/**
 * \brief finds bitset per polyline with colliding edges
 * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
 */
MRMESH_API std::pair<UndirectedEdgeBitSet, UndirectedEdgeBitSet> findCollidingEdgesBitsets( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr );

/// finds all pairs of colliding edges from 2d polyline
MRMESH_API std::vector<EdgePointPair> findSelfCollidingEdgePairs( const Polyline2& polyline );

/// finds all pairs of colliding edges from 2d polyline
MRMESH_API std::vector<UndirectedEdgeUndirectedEdge> findSelfCollidingEdges( const Polyline2& polyline );

/// finds the union of all self-intersecting edges
MRMESH_API UndirectedEdgeBitSet findSelfCollidingEdgesBS( const Polyline2& polyline );

/**
 * \brief checks that arbitrary 2d polyline A is inside of closed 2d polyline B
 * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
 */
MRMESH_API bool isInside( const Polyline2& a, const Polyline2& b,
    const AffineXf2f* rigidB2A = nullptr );

/// \}

} // namespace MR