#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// stores reference on whole mesh (if region is NULL) or on its part (if region pointer is valid)
typedef struct MRMeshPart
{
    const MRMesh* mesh;
    const MRFaceBitSet* region;
} MRMeshPart;

MR_EXTERN_C_END
