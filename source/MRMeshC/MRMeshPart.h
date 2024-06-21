#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRMESHC_CLASS MRMeshPart
{
    const MRMesh* mesh;
    const MRFaceBitSet* region;
} MRMeshPart;

MR_EXTERN_C_END
