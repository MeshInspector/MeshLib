#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

MRMESHC_API const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

MRMESHC_API const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

MRMESHC_API MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top );

MRMESHC_API const MREdgeId* mrEdgePathData( const MREdgePath* ep );

MRMESHC_API size_t mrEdgePathSize( const MREdgePath* ep );

MRMESHC_API void mrEdgePathFree( MREdgePath* ep );

MR_EXTERN_C_END
