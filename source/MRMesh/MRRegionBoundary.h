#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

namespace MR
{

// returns closed loop of region boundary starting from given region boundary edge;
// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to right
[[nodiscard]] MRMESH_API EdgeLoop trackRegionBoundaryLoop( const MeshTopology & topology, EdgeId e0, const FaceBitSet * region = nullptr );
[[nodiscard]] inline EdgeLoop trackRegionBoundaryLoop( const MeshTopology & topology, const FaceBitSet & region, EdgeId e0 )
    { return trackRegionBoundaryLoop( topology, e0, &region ); }

// returns all region boundary loops;
// every loop has region faces on the left, and not-region faces or holes on the right
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> findRegionBoundary( const MeshTopology & topology, const FaceBitSet * region = nullptr );
[[nodiscard]] inline std::vector<EdgeLoop> findRegionBoundary( const MeshTopology & topology, const FaceBitSet & region )
    { return findRegionBoundary( topology, &region ); }

// returns all region boundary paths;
// every path has region faces on the left, and valid not-region faces on the right
[[nodiscard]] MRMESH_API std::vector<EdgePath> findRegionBoundaryInsideMesh( const MeshTopology & topology, const FaceBitSet & region );

// returns all region boundary edges, where each edge has a region face on one side, and a valid not-region face on another side
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findRegionBoundaryUndirectedEdgesInsideMesh( const MeshTopology & topology, const FaceBitSet & region );

/// \returns All out of region faces that have a common edge with at least one region face
[[nodiscard]] MRMESH_API FaceBitSet findRegionOuterFaces( const MeshTopology& topology, const FaceBitSet& region );

// composes the set of all vertices incident to given faces
[[nodiscard]] MRMESH_API VertBitSet getIncidentVerts( const MeshTopology & topology, const FaceBitSet & faces );
// if faces-parameter is null pointer then simply returns the reference on all valid vertices;
// otherwise performs store = getIncidentVerts( topology, *faces ) and returns reference on store
[[nodiscard]] MRMESH_API const VertBitSet & getIncidentVerts( const MeshTopology & topology, const FaceBitSet * faces, VertBitSet & store );
// composes the set of all vertices with all their faces in given set
[[nodiscard]] MRMESH_API VertBitSet getInnerVerts( const MeshTopology & topology, const FaceBitSet & faces );
// composes the set of all boundary vertices for given region (or whole mesh if !region)
[[nodiscard]] MRMESH_API VertBitSet getBoundaryVerts( const MeshTopology & topology, const FaceBitSet * region = nullptr );

// composes the set of all faces incident to given vertices
[[nodiscard]] MRMESH_API FaceBitSet getIncidentFaces( const MeshTopology & topology, const VertBitSet & verts );
// composes the set of all faces with all their vertices in given set
[[nodiscard]] MRMESH_API FaceBitSet getInnerFaces( const MeshTopology & topology, const VertBitSet & verts );

// composes the set of all edges, having a face from given set at the left
[[nodiscard]] MRMESH_API EdgeBitSet getRegionEdges( const MeshTopology& topology, const FaceBitSet& faces );
// composes the set of all undirected edges, having a face from given set from one of two sides
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getIncidentEdges( const MeshTopology& topology, const FaceBitSet& faces );
// composes the set of all vertices incident to given edges
[[nodiscard]] MRMESH_API VertBitSet getIncidentVerts( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );
// composes the set of all faces incident to given edges
[[nodiscard]] MRMESH_API FaceBitSet getIncidentFaces( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );
// composes the set of all edges with all their vertices in given set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getInnerEdges( const MeshTopology & topology, const VertBitSet& verts );
// composes the set of all edges having both left and right in given region
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getInnerEdges( const MeshTopology & topology, const FaceBitSet& region );
// if edges-parameter is null pointer then simply returns the reference on all valid vertices;
// otherwise performs store = getIncidentVerts( topology, *edges ) and returns reference on store
[[nodiscard]] MRMESH_API const VertBitSet & getIncidentVerts( const MeshTopology & topology, const UndirectedEdgeBitSet * edges, VertBitSet & store );
// composes the set of all vertices with all their edges in given set
[[nodiscard]] MRMESH_API VertBitSet getInnerVerts( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );

} //namespace MR
