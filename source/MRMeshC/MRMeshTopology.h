#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

MRMESHC_API const MRBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

MRMESHC_API const MRBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

typedef struct MRTriangulation MRTriangulation;

MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

#ifdef __cplusplus
}
#endif
