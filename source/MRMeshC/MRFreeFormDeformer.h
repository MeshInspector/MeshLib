#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN
// Class for deforming mesh using trilinear interpolation
typedef struct MRFreeFormDeformer MRFreeFormDeformer;

MRMESHC_API MRFreeFormDeformer* mrFreeFormDeformerNewFromMesh( MRMesh* mesh, const MRVertBitSet* region );

MRMESHC_API void mrFreeFormDeformerFree( MRFreeFormDeformer* deformer );

// Parallel calculates all points normed positions
// sets ref grid by initialBox, if initialBox is invalid use mesh bounding box instead
MRMESHC_API void mrFreeFormDeformerInit( MRFreeFormDeformer* deformer, const MRVector3i* resolution, const MRBox3f* initialBox );

// Updates ref grid point position
MRMESHC_API void mrFreeFormDeformerSetRefGridPointPosition( MRFreeFormDeformer* deformer, const MRVector3i* coordOfPointInGrid, const MRVector3f* newPos );

// Gets ref grid point position
MRMESHC_API MRVector3f mrFreeFormDeformerGetRefGridPointPosition( const MRFreeFormDeformer* deformer, const MRVector3i* coordOfPointInGrid );

// Parallel apply updated grid to all mesh points
// ensure updating render object after using it
MRMESHC_API void mrFreeFormDeformerApply( const MRFreeFormDeformer* deformer );

MR_EXTERN_C_END
