#pragma once

#include "MRId.h"
#include "MRBuffer.h"
#include <tuple>

namespace MR
{

/// compute the order of vertices given the order of faces:
/// vertices near first faces also appear first;
/// \param faceMap old face id -> new face id
[[nodiscard]] MRMESH_API VertBMap getVertexOrdering( const FaceBMap & faceMap, const MeshTopology & topology );

/// compute the order of edges given the order of faces:
/// edges near first faces also appear first;
/// \param faceMap old face id -> new face id
[[nodiscard]] MRMESH_API UndirectedEdgeBMap getEdgeOrdering( const FaceBMap & faceMap, const MeshTopology & topology );

} //namespace MR
