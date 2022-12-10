#pragma once

#include "MRId.h"
#include "MRBuffer.h"
#include <tuple>

namespace MR
{

struct OrderedVertex
{
    VertId v;
    std::uint32_t f; // the smallest nearby face
    bool operator <( const OrderedVertex & b ) const
        { return std::tie( f, v ) < std::tie( b.f, b.v ); } // order vertices by f
};
static_assert( sizeof( OrderedVertex ) == 8 );

/// compute the order of vertices given the order of faces:
/// vertices near first faces also appear first;
/// \param invFaceMap old face id -> new face id
[[nodiscard]] MRMESH_API VertexOrdering getVertexOrdering( const Buffer<FaceId> & invFaceMap, const MeshTopology & topology );

struct OrderedEdge
{
    UndirectedEdgeId ue;
    std::uint32_t f; // the smallest nearby face
    bool operator <( const OrderedEdge & b ) const
        { return std::tie( f, ue ) < std::tie( b.f, b.ue ); } // order vertices by f
};
static_assert( sizeof( OrderedEdge ) == 8 );

/// compute the order of edges given the order of faces:
/// edges near first faces also appear first;
/// \param invFaceMap old face id -> new face id
[[nodiscard]] MRMESH_API EdgeOrdering getEdgeOrdering( const Buffer<FaceId> & invFaceMap, const MeshTopology & topology );

} //namespace MR
