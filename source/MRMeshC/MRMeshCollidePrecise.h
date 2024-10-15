#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshPart.h"
#include "MRPrecisePredicates3.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// edge from one mesh and triangle from another mesh
typedef struct MREdgeTri
{
    MREdgeId edge;
    MRFaceId tri;
} MREdgeTri;

MRMESHC_API bool mrEdgeTriEq( const MREdgeTri* a, const MREdgeTri* b );

MR_VECTOR_DECL( EdgeTri )

typedef struct MRPreciseCollisionResult MRPreciseCollisionResult;

/// each edge is directed to have its origin inside and its destination outside of the other mesh
MRMESHC_API const MRVectorEdgeTri mrPreciseCollisionResultEdgesAtrisB( const MRPreciseCollisionResult* result );

/// each edge is directed to have its origin inside and its destination outside of the other mesh
MRMESHC_API const MRVectorEdgeTri mrPreciseCollisionResultEdgesBtrisA( const MRPreciseCollisionResult* result );

/// deallocates the PreciseCollisionResult object
MRMESHC_API void mrPreciseCollisionResultFree( MRPreciseCollisionResult* result );

/**
 * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, NULL considered as identity transformation
 * \param anyIntersection if true then the function returns as fast as it finds any intersection
 */
MRMESHC_API MRPreciseCollisionResult* mrFindCollidingEdgeTrisPrecise( const MRMeshPart* a, const MRMeshPart* b,
                                                                      const MRConvertToIntVector* conv,
                                                                      const MRAffineXf3f* rigidB2A,
                                                                      bool anyIntersection );

/**
 * \brief creates simple converters from Vector3f to Vector3i and back in mesh parts area range
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESHC_API MRCoordinateConverters mrGetVectorConverters( const MRMeshPart* a, const MRMeshPart* b,
                                                          const MRAffineXf3f* rigidB2A );

MR_EXTERN_C_END
