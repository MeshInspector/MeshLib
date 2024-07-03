#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

/// tightly packs all arrays eliminating lone edges and invalid faces and vertices
MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

/// returns cached set of all valid vertices
MRMESHC_API const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

/// returns cached set of all valid faces
MRMESHC_API const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

/// returns three vertex ids for valid triangles (which can be accessed by FaceId),
/// vertex ids for invalid triangles are undefined, and shall not be read
MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

/// gets read-only access to the vertex triples of the triangulation
MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

/// gets total count of the vertex triples of the triangulation
MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

/// deallocates the Triangulation object
MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

/// returns one edge with no valid left face for every boundary in the mesh
MRMESHC_API MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top );

/// gets read-only access to the edges of the edge path
MRMESHC_API const MREdgeId* mrEdgePathData( const MREdgePath* ep );

/// gets total count of the edges of the edge path
MRMESHC_API size_t mrEdgePathSize( const MREdgePath* ep );

/// deallocates the EdgePath object
MRMESHC_API void mrEdgePathFree( MREdgePath* ep );

MR_EXTERN_C_END
