#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshPart.h"
#include "MRPrecisePredicates3.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MREdgeTri
{
    MREdgeId edge;
    MRFaceId tri;
} MREdgeTri;

/// ...
MRMESHC_API bool mrEdgeTriEq( const MREdgeTri* a, const MREdgeTri* b );

MR_VECTOR_DECL( EdgeTri )

/// ...
typedef struct MRPreciseCollisionResult
{
    MRVectorEdgeTri* edgesAtrisB;
    MRVectorEdgeTri* edgesBtrisA;
} MRPreciseCollisionResult;

/// ...
MRMESHC_API MRPreciseCollisionResult mrFindCollidingEdgeTrisPrecise( const MRMeshPart* a, const MRMeshPart* b,
                                                                     const MRConvertToIntVector* conv,
                                                                     const MRAffineXf3f* rigidB2A,
                                                                     bool anyIntersection );

/// ...
MRMESHC_API MRCoordinateConverters mrGetVectorConverters( const MRMeshPart* a, const MRMeshPart* b,
                                                          const MRAffineXf3f* rigidB2A );

MR_EXTERN_C_END
