#pragma once

#include "MRId.h"
#include <compare>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// a pair of faces
struct FaceFace
{
    FaceId aFace;
    FaceId bFace;
    FaceFace( FaceId a, FaceId b ) : aFace( a ), bFace( b ) {}
    FaceFace() {};
    auto operator<=>( const FaceFace& rhs ) const = default;
};

/// a pair of undirected edges
struct UndirectedEdgeUndirectedEdge
{
    UndirectedEdgeId aUndirEdge;
    UndirectedEdgeId bUndirEdge;
    UndirectedEdgeUndirectedEdge( UndirectedEdgeId a, UndirectedEdgeId b ) : aUndirEdge( a ), bUndirEdge( b ) {}
    UndirectedEdgeUndirectedEdge() {};
    auto operator<=>( const UndirectedEdgeUndirectedEdge& rhs ) const = default;
};

/// \}

} // namespace MR
