#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRNoiseSettings
{
    float sigma;
    // start state of the generator engine
    unsigned int seed;
    MRProgressCallback callback;
} MRNoiseSettings;

MRMESHC_API MRNoiseSettings mrNoiseSettingsNew( void );

// Adds noise to the points, using a normal distribution
MRMESHC_API void mrAddNoiseToMesh( MRMesh* mesh, const MRVertBitSet* region, const MRNoiseSettings* settings, MRString** errorString );

MR_EXTERN_C_END
