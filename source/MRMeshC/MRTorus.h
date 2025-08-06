#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// parameters for \ref mrMakeTorus
typedef struct MRMakeTorusParameters
{
    float primaryRadius;
    float secondaryRadius;
    int primaryResolution;
    int secondaryResolution;
    // TODO: points
} MRMakeTorusParameters;

/// initializes a default instance
MRMESHC_API MRMakeTorusParameters mrMakeTorusParametersNew( void );

/// creates a mesh representing a torus
/// Z is symmetry axis of this torus
MRMESHC_API MRMesh* mrMakeTorus( const MRMakeTorusParameters* params );
// creates torus with empty sectors
// main application - testing Components
MRMESHC_API MRMesh* mrMakeTorusWithSelfIntersections( const MRMakeTorusParameters* params );

MR_EXTERN_C_END
