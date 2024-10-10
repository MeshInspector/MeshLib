#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRTriPoint.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRMeshTriPoint
{
    /// ...
    MREdgeId e;
    /// ...
    MRTriPointf bary;
} MRMeshTriPoint;

MR_EXTERN_C_END
