#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MREdgeId { int id; } MREdgeId;
typedef struct MRFaceId { int id; } MRFaceId;
typedef struct MRVertId { int id; } MRVertId;

typedef MRVertId MRThreeVertIds[3];

MR_EXTERN_C_END
