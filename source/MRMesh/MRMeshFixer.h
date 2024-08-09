#pragma once

#include "MRId.h"
#include "MRProgressCallback.h"
#include <cfloat>
#include "MRExpected.h"
#include <string>

namespace MR
{

/// \defgroup MeshFixerGroup Mesh Fixer
/// \ingroup MeshAlgorithmGroup
/// \{

/// Duplicates all vertices having more than two boundary edges (and returns the number of duplications);
MRMESH_API int duplicateMultiHoleVertices( Mesh & mesh );

/// finds multiple edges in the mesh
using MultipleEdge = VertPair;
[[nodiscard]] MRMESH_API Expected<std::vector<MultipleEdge>> findMultipleEdges( const MeshTopology & topology, ProgressCallback cb = {} );
[[nodiscard]] inline bool hasMultipleEdges( const MeshTopology & topology ) { return !findMultipleEdges( topology ).value().empty(); }

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESH_API void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges );
/// finds and resolves multiple edges
MRMESH_API void fixMultipleEdges( Mesh & mesh );

/// finds faces having aspect ratio >= criticalAspectRatio
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio = FLT_MAX, ProgressCallback cb = {} );

/// finds edges having length <= criticalLength
[[nodiscard]] MRMESH_API Expected<UndirectedEdgeBitSet> findShortEdges( const MeshPart& mp, float criticalLength, ProgressCallback cb = {} );

/// finds vertices in region with complete ring of N edges
[[nodiscard]] MRMESH_API VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region = nullptr );

/// returns true if the edge e has both left and right triangular faces and the degree of dest( e ) is 2
[[nodiscard]] MRMESH_API bool isEdgeBetweenDoubleTris( const MeshTopology& topology, EdgeId e );

/// if the edge e has both left and right triangular faces and the degree of dest( e ) is 2,
/// then eliminates left( e ), right( e ), e, e.sym(), next( e ), dest( e ), and returns prev( e );
/// if region is provided then eliminated faces are excluded from it;
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDoubleTris( MeshTopology& topology, EdgeId e, FaceBitSet * region = nullptr );

/// eliminates all double triangles around given vertex preserving vertex valid;
/// if region is provided then eliminated triangles are excluded from it
MRMESH_API void eliminateDoubleTrisAround( MeshTopology & topology, VertId v, FaceBitSet * region = nullptr );

/// returns true if the destination of given edge has degree 3 and 3 incident triangles
[[nodiscard]] MRMESH_API bool isDegree3Dest( const MeshTopology& topology, EdgeId e );

/// if the destination of given edge has degree 3 and 3 incident triangles,
/// then eliminates the destination vertex with all its edges and all but one faces, and returns valid remaining edge with same origin as e;
/// if region is provided then eliminated triangles are excluded from it;
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDegree3Dest( MeshTopology& topology, EdgeId e, FaceBitSet * region = nullptr );

/// eliminates from the mesh all vertices having degree 3 and 3 incident triangles from given region (which is updated);
/// if \param fs is provided then eliminated triangles are excluded from it;
/// \return the number of vertices eliminated
MRMESH_API int eliminateDegree3Vertices( MeshTopology& topology, VertBitSet & region, FaceBitSet * fs = nullptr );

/// if given vertex is present on the boundary of some hole several times then returns an edge of this hole (without left);
/// returns invalid edge otherwise (not a boundary vertex, or it is present only once on the boundary of each hole it pertains to)
[[nodiscard]] MRMESH_API EdgeId isVertexRepeatedOnHoleBd( const MeshTopology& topology, VertId v );

/// returns set bits for all vertices present on the boundary of a hole several times;
[[nodiscard]] MRMESH_API VertBitSet findRepeatedVertsOnHoleBd( const MeshTopology& topology );

/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
[[nodiscard]] MRMESH_API FaceBitSet findHoleComplicatingFaces( const Mesh & mesh );

/// \}

} // namespace MR
