#pragma once

#include "MRId.h"
#include "MRBuffer.h"
#include <tuple>

namespace MR
{

/// computes optimal order of faces: old face id -> new face id,
/// the order is similar as in AABB tree, but faster to compute
[[nodiscard]] MRMESH_API FaceBMap getOptimalFaceOrdering( const Mesh & mesh );

/// compute the order of vertices given the order of faces:
/// vertices near first faces also appear first;
/// \param faceMap old face id -> new face id
[[nodiscard]] MRMESH_API VertBMap getVertexOrdering( const FaceBMap & faceMap, const MeshTopology & topology );

/// compute the order of edges given the order of faces:
/// edges near first faces also appear first;
/// \param faceMap old face id -> new face id
[[nodiscard]] MRMESH_API UndirectedEdgeBMap getEdgeOrdering( const FaceBMap & faceMap, const MeshTopology & topology );

} //namespace MR
