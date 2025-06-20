#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// tightly packs all arrays eliminating lone edges and invalid faces and vertices
MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

/// returns cached set of all valid vertices
MRMESHC_API const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

/// returns cached set of all valid faces
MRMESHC_API const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

MR_VECTOR_LIKE_DECL( Triangulation, ThreeVertIds )

/// returns three vertex ids for valid triangles (which can be accessed by FaceId),
/// vertex ids for invalid triangles are undefined, and shall not be read
MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

/// returns the number of face records including invalid ones
MRMESHC_API size_t mrMeshTopologyFaceSize( const MRMeshTopology* top );

MR_VECTOR_LIKE_DECL( EdgePath, EdgeId )
typedef MREdgePath MREdgeLoop;

/// returns one edge with no valid left face for every boundary in the mesh
MRMESHC_API MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top );

/// gets 3 vertices of given triangular face;
/// the vertices are returned in counter-clockwise order if look from mesh outside
MRMESHC_API void mrMeshTopologyGetLeftTriVerts( const MRMeshTopology* top, MREdgeId a, MRVertId* v0, MRVertId* v1, MRVertId* v2 );

/// gets 3 vertices of given triangular face;
/// the vertices are returned in counter-clockwise order if look from mesh outside
MRMESHC_API void mrMeshTopologyGetTriVerts( const MRMeshTopology* top, MRFaceId f, MRVertId* v0, MRVertId* v1, MRVertId* v2 );

/// returns the number of hole loops in the mesh;
/// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
MRMESHC_API int mrMeshTopologyFindNumHoles( const MRMeshTopology* top, MREdgeBitSet* holeRepresentativeEdges );

MR_EXTERN_C_END
