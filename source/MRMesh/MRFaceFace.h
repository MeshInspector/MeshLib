#pragma once

#include "MRId.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct FaceFace
{
    FaceId aFace;
    FaceId bFace;
    FaceFace( FaceId a, FaceId b ) : aFace( a ), bFace( b ) { }
    FaceFace() { };
    bool operator==( const FaceFace& rhs ) const = default;
};

struct UndirectedEdgeUndirectedEdge
{
    UndirectedEdgeId aUndirEdge;
    UndirectedEdgeId bUndirEdge;
    UndirectedEdgeUndirectedEdge( UndirectedEdgeId a, UndirectedEdgeId b ) : aUndirEdge( a ), bUndirEdge( b )
    {}
    UndirectedEdgeUndirectedEdge()
    {};
    bool operator==( const UndirectedEdgeUndirectedEdge& rhs ) const = default;
};

/// \}

} // namespace MR
