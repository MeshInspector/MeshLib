#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

MRMESHC_API void mrExpandFaceRegion( const MRMeshTopology* top, MRFaceBitSet* region, int hops );

MRMESHC_API MRFaceBitSet* mrExpandFaceRegionFromFace( const MRMeshTopology* top, MRFaceId face, int hops );

MRMESHC_API void mrExpandVertRegion( const MRMeshTopology* top, MRVertBitSet* region, int hops );

MRMESHC_API MRVertBitSet* mrExpandVertRegionFromVert( const MRMeshTopology* top, MRVertId vert, int hops );

MRMESHC_API void mrShrinkFaceRegion( const MRMeshTopology* top, MRFaceBitSet* region, int hops );

MRMESHC_API void mrShrinkVertRegion( const MRMeshTopology* top, MRVertBitSet* region, int hops );

MR_EXTERN_C_END
