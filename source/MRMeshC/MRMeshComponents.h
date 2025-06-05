#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshPart.h"
#include "MRVector.h"


MR_EXTERN_C_BEGIN

MR_VECTOR_LIKE_DECL( Face2RegionMap, RegionId )


/// stores reference on Face2RegionMap ( key: face id, value: region id ) and number of components
typedef struct MRMeshComponentsMap
{
    MRFace2RegionMap* faceMap;
    int numComponents;
} MRMeshComponentsMap;

/// stores reference on bitset of faces and number of regions
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

/// returns one connected component containing given face, 
/// not effective to call more than once, if several components are needed use getAllComponents
MRMESHC_API MRFaceBitSet* mrMeshComponentsGetComponent( const MRMeshPart* mp, MRFaceId id, MRFaceIncidence incidence, const MRUndirectedEdgeBitSet* isCompBd );
/// returns the largest by surface area component or empty set if its area is smaller than \param minArea
MRMESHC_API MRFaceBitSet* mrMeshComponentsGetLargestComponent( const MRMeshPart* mp, MRFaceIncidence incidence, const MRUndirectedEdgeBitSet* isCompBd, float minArea, int* numSmallerComponents );
/// returns the union of connected components, each having at least given area
MRMESHC_API MRFaceBitSet* mrMeshComponentsGetLargeByAreaComponents( const MRMeshPart* mp, float minArea, const MRUndirectedEdgeBitSet* isCompBd );
/// gets all connected components of mesh part as
/// 1. the mapping: FaceId -> Component ID in [0, 1, 2, ...)
/// 2. the total number of components
MRMESHC_API MRMeshComponentsMap mrMeshComponentsGetAllComponentsMap( const MRMeshPart* mp, MRFaceIncidence incidence );

/// returns
/// 1. the union of all regions with area >= minArea
/// 2. the number of such regions
MRMESHC_API MRMeshRegions mrMeshComponentsGetLargeByAreaRegions( const MRMeshPart* mp, const MRFace2RegionMap* face2RegionMap, int numRegions, float minArea );

/// deletes allocated map
MRMESHC_API void mrMeshComponentsAllComponentsMapFree( const MRMeshComponentsMap* map );

MR_EXTERN_C_END
