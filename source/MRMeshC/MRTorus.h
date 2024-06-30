#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRMakeTorusParameters
{
    float primaryRadius;
    float secondaryRadius;
    int primaryResolution;
    int secondaryResolution;
    // TODO: points
} MRMakeTorusParameters;

MRMESHC_API MRMakeTorusParameters mrMakeTorusParametersDefault();

MRMESHC_API MRMesh* mrMakeTorus( const MRMakeTorusParameters* params );

MR_EXTERN_C_END