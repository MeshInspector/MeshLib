#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

/// Duplicates all vertices having more than two boundary edges (and returns the number of duplications);
MRMESH_API int duplicateMultiHoleVertices( Mesh & mesh );

// find multiple edges in the mesh
using MultipleEdge = std::pair<VertId, VertId>;
MRMESH_API std::vector<MultipleEdge> findMultipleEdges( const MeshTopology & topology );

// finds faces which aspect ratio >= criticalAspectRatio
MRMESH_API FaceBitSet findDegenerateFaces( const Mesh& mesh, float criticalAspectRatio = FLT_MAX );

} //namespace MR
