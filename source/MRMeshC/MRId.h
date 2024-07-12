#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// edge index
typedef struct MREdgeId { int id; } MREdgeId;
/// face index
typedef struct MRFaceId { int id; } MRFaceId;
/// vertex index
typedef struct MRVertId { int id; } MRVertId;

/// a set of 3 vertices; useful for representing a face via its vertex indices
typedef MRVertId MRThreeVertIds[3];

MR_EXTERN_C_END
