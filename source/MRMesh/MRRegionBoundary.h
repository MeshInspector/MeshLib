#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

namespace MR
{

// returns closed loop of region boundary starting from given region boundary edge;
// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to right
MRMESH_API EdgeLoop trackRegionBoundaryLoop( const MeshTopology & topology, EdgeId e0, const FaceBitSet * region = nullptr );
inline EdgeLoop trackRegionBoundaryLoop( const MeshTopology & topology, const FaceBitSet & region, EdgeId e0 )
    { return trackRegionBoundaryLoop( topology, e0, &region ); }

// returns all region boundary loops;
// every loop has region faces on the left, and not-region faces or holes on the right
MRMESH_API std::vector<EdgeLoop> findRegionBoundary( const MeshTopology & topology, const FaceBitSet * region = nullptr );
inline std::vector<EdgeLoop> findRegionBoundary( const MeshTopology & topology, const FaceBitSet & region )
    { return findRegionBoundary( topology, &region ); }

/// \returns All out of region faces that have a common edge with at least one region face
MRMESH_API FaceBitSet findRegionOuterFaces( const MeshTopology& topology, const FaceBitSet& region );

// composes the set of all vertices incident to given faces
MRMESH_API VertBitSet getIncidentVerts( const MeshTopology & topology, const FaceBitSet & faces );
// if faces-parameter is null pointer then simply returns the reference on all valid vertices;
// otherwise performs store = getIncidentVerts( topology, *faces ) and returns reference on store
MRMESH_API const VertBitSet & getIncidentVerts( const MeshTopology & topology, const FaceBitSet * faces, VertBitSet & store );
// composes the set of all vertices with all their faces in given set
MRMESH_API VertBitSet getInnerVerts( const MeshTopology & topology, const FaceBitSet & faces );
// composes the set of all boundary vertices for given region (or whole mesh if !region)
MRMESH_API VertBitSet getBoundaryVerts( const MeshTopology & topology, const FaceBitSet * region = nullptr );

// composes the set of all faces incident to given vertices
MRMESH_API FaceBitSet getIncidentFaces( const MeshTopology & topology, const VertBitSet & verts );
// composes the set of all faces with all their vertices in given set
MRMESH_API FaceBitSet getInnerFaces( const MeshTopology & topology, const VertBitSet & verts );

// composes the set of all edges, all vertices of which are incident to given set
MRMESH_API EdgeBitSet getRegionEdges( const MeshTopology& topology, const FaceBitSet& faces );
// composes the set of all vertices incident to given edges
MRMESH_API VertBitSet getIncidentVerts( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );
// composes the set of all faces incident to given edges
MRMESH_API FaceBitSet getIncidentFaces( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );
// composes the set of all edges with all their vertices in given set
MRMESH_API UndirectedEdgeBitSet getInnerEdges( const MeshTopology & topology, const VertBitSet& verts );
// if edges-parameter is null pointer then simply returns the reference on all valid vertices;
// otherwise performs store = getIncidentVerts( topology, *edges ) and returns reference on store
MRMESH_API const VertBitSet & getIncidentVerts( const MeshTopology & topology, const UndirectedEdgeBitSet * edges, VertBitSet & store );
// composes the set of all vertices with all their edges in given set
MRMESH_API VertBitSet getInnerVerts( const MeshTopology & topology, const UndirectedEdgeBitSet & edges );

} //namespace MR
