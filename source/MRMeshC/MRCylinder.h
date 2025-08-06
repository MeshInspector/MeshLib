#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// optional parameters for \ref mrMakeCylinderAdvanced
typedef struct MRMakeCylinderAdvancedParameters
{
    float radius0;
    float radius1;
    float startAngle;
    float arcSize;
    float length;
    int resolution;
} MRMakeCylinderAdvancedParameters;

/// initializes a default instance
MRMESHC_API MRMakeCylinderAdvancedParameters mrMakeCylinderAdvancedParametersNew( void );

// creates a mesh representing a cylinder
MRMESHC_API MRMesh* mrMakeCylinderAdvanced( const MRMakeCylinderAdvancedParameters* params );

MR_EXTERN_C_END
