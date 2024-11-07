#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshPart.h"
#include "MRVector.h"


MR_EXTERN_C_BEGIN

MR_VECTOR_LIKE_DECL( Face2RegionMap, RegionId )

typedef struct MRMeshComponentsMap
{
    MRFace2RegionMap* faceMap;
    int numComponents;
} MRMeshComponentsMap;

typedef struct MRMeshRegions
{
    MRFaceBitSet* faces;
    int numRegions;
} MRMeshRegions;

typedef enum MRFaceIncidence
{
    MRFaceIncidencePerEdge, ///< face can have neighbor only via edge
    MRFaceIncidencePerVertex ///< face can have neighbor via vertex
} MRFaceIncidence;

MRMESHC_API MRMeshComponentsMap mrMeshComponentsGetAllComponentsMap( const MRMeshPart* mp, MRFaceIncidence incidence );

MRMESHC_API MRMeshRegions mrMeshComponentsGetLargeByAreaRegions( const MRMeshPart* mp, const MRFace2RegionMap* face2RegionMap, int numRegions, float minArea );

MRMESHC_API void mrMeshComponentsAllComponentsMapFree( const MRMeshComponentsMap* map );

MR_EXTERN_C_END
