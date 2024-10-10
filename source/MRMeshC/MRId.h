#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// edge index
typedef struct MREdgeId { int id; } MREdgeId;
/// edge index
typedef struct MRUndirectedEdgeId { int id; } MRUndirectedEdgeId;
/// face index
typedef struct MRFaceId { int id; } MRFaceId;
/// vertex index
typedef struct MRVertId { int id; } MRVertId;
/// object index
typedef struct MRObjId { int id; } MRObjId;

/// a set of 3 vertices; useful for representing a face via its vertex indices
typedef MRVertId MRThreeVertIds[3];

/// ...
MRMESHC_API MREdgeId mrEdgeIdFromUndirectedEdgeId( MRUndirectedEdgeId u );

/// ...
MRMESHC_API MREdgeId mrEdgeIdSym( MREdgeId e );

/// ...
MRMESHC_API MRUndirectedEdgeId mrEdgeIdUndirected( MREdgeId e );

MR_EXTERN_C_END
