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
[[nodiscard]] MRMESH_API std::vector<MultipleEdge> findMultipleEdges( const MeshTopology & topology );
[[nodiscard]] inline bool hasMultipleEdges( const MeshTopology & topology ) { return !findMultipleEdges( topology ).empty(); }

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESH_API void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges );
/// finds and resolves multiple edges
MRMESH_API void fixMultipleEdges( Mesh & mesh );

/// finds faces having aspect ratio >= criticalAspectRatio
[[nodiscard]] MRMESH_API FaceBitSet findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio = FLT_MAX );

/// finds edges having length <= criticalLength
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findShortEdges( const MeshPart& mp, float criticalLength );

/// finds vertices in region with complete ring of N edges
[[nodiscard]] MRMESH_API VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region = nullptr );

/// returns true if the edge e has both left and right triangular faces and the degree of dest( e ) is 2
[[nodiscard]] MRMESH_API bool isEdgeBetweenDoubleTris( const MeshTopology& topology, EdgeId e );

/// if the edge e has both left and right triangular faces and the degree of dest( e ) is 2,
/// then eliminates left( e ), right( e ), e, e.sym(), next( e ), dest( e ), and returns prev( e );
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDoubleTris( MeshTopology& topology, EdgeId e );

///  eliminates all double triangles around given vertex preserving vertex valid
MRMESH_API void eliminateDoubleTrisAround( MeshTopology & topology, VertId v );

/// returns true if the destination of given edge has degree 3 and 3 incident triangles
[[nodiscard]] MRMESH_API bool isDegree3Dest( const MeshTopology& topology, EdgeId e );

/// if the destination of given edge has degree 3 and 3 incident triangles,
/// then eliminates the destination vertex with all its edges and all but one faces, and returns valid remaining edge with same origin as e;
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDegree3Dest( MeshTopology& topology, EdgeId e );

/// eliminates from the mesh all vertices having degree 3 and 3 incident triangles from given region (which is updated);
/// \return the number of vertices eliminated
MRMESH_API int eliminateDegree3Vertices( MeshTopology& topology, VertBitSet & region );

/// \}

} // namespace MR
