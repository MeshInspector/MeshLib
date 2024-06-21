#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

MRMESHC_API const MRBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

MRMESHC_API const MRBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

MR_EXTERN_C_END
