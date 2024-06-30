#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRMeshPart
{
    const MRMesh* mesh;
    const MRFaceBitSet* region;
} MRMeshPart;

MR_EXTERN_C_END
