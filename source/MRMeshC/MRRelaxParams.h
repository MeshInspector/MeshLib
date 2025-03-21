#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRRelaxParams
{
    /// number of iterations
    int iterations;

    /// region to relax
    const MRVertBitSet *region;

    /// speed of relaxing, typical values (0.0, 0.5]
    float force;

    /// if true then maximal displacement of each point during denoising will be limited
    bool limitNearInitial;

    /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
    float maxInitialDist;
} MRRelaxParams;

MRMESHC_API MRRelaxParams mrRelaxParamsNew( void );

MR_EXTERN_C_END
