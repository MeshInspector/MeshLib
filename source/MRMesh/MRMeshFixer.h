#pragma once

#include "MRId.h"
#include <cfloat>

namespace MR
{

/// \defgroup MeshFixerGroup Mesh Fixer
/// \ingroup MeshAlgorithmGroup
/// \{

/// Duplicates all vertices having more than two boundary edges (and returns the number of duplications);
MRMESH_API int duplicateMultiHoleVertices( Mesh & mesh );

/// finds multiple edges in the mesh
using MultipleEdge = std::pair<VertId, VertId>;
MRMESH_API std::vector<MultipleEdge> findMultipleEdges( const MeshTopology & topology );
inline bool hasMultipleEdges( const MeshTopology & topology ) { return !findMultipleEdges( topology ).empty(); }

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESH_API void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges );
/// finds and resolves multiple edges
MRMESH_API void fixMultipleEdges( Mesh & mesh );

/// finds faces having aspect ratio >= criticalAspectRatio
MRMESH_API FaceBitSet findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio = FLT_MAX );

/// finds edges having length <= criticalLength
MRMESH_API UndirectedEdgeBitSet findShortEdges( const MeshPart& mp, float criticalLength );

/// finds vertices in region with complete ring of N edges
MRMESH_API VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region = nullptr );

/// \}

} // namespace MR
