#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mrMeshTopologyPack( MRMeshTopology* top );

const MRBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

const MRBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

typedef struct MRTriangulation MRTriangulation;

MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

size_t mrTriangulationSize( const MRTriangulation* tris );

void mrTriangulationFree( MRTriangulation* tris );

#ifdef __cplusplus
}
#endif
