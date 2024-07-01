#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRMakeCylinderAdvancedParameters
{
    float radius0;
    float radius1;
    float startAngle;
    float arcSize;
    float length;
    int resolution;
} MRMakeCylinderAdvancedParameters;

MRMESHC_API MRMakeCylinderAdvancedParameters mrMakeCylinderAdvancedParametersDefault( void );

MRMESHC_API MRMesh* mrMakeCylinderAdvanced( const MRMakeCylinderAdvancedParameters* params );

MR_EXTERN_C_END