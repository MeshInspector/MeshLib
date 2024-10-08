#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// parameters for \ref mrMakeSphere
typedef struct MRSphereParams
{
    float radius;
    int numMeshVertices;
} MRSphereParams;

/// initializes a default instance
MRMESHC_API MRSphereParams mrSphereParamsNew( void );

/// creates a mesh of sphere with irregular triangulation
MRMESHC_API MRMesh* mrMakeSphere( const MRSphereParams* params );

/// parameters for \ref mrMakeUVSphere
typedef struct MRMakeUVSphereParameters
{
    float radius;
    int horizontalResolution;
    int verticalResolution;
} MRMakeUVSphereParameters;

/// initializes a default instance
MRMESHC_API MRMakeUVSphereParameters mrMakeUvSphereParametersNew( void );

/// creates a mesh of sphere with regular triangulation (parallels and meridians)
MRMESHC_API MRMesh* mrMakeUVSphere( const MRMakeUVSphereParameters* params );

MR_EXTERN_C_END
