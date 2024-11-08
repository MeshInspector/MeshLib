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
/// region index
typedef struct MRRegionId { int id; } MRRegionId;

/// a set of 3 vertices; useful for representing a face via its vertex indices
typedef MRVertId MRThreeVertIds[3];

/// creates an edge id from the corresponding undirected one
MRMESHC_API MREdgeId mrEdgeIdFromUndirectedEdgeId( MRUndirectedEdgeId u );

// returns identifier of the edge with same ends but opposite orientation
MRMESHC_API MREdgeId mrEdgeIdSym( MREdgeId e );

// returns unique identifier of the edge ignoring its direction
MRMESHC_API MRUndirectedEdgeId mrEdgeIdUndirected( MREdgeId e );

MR_EXTERN_C_END
