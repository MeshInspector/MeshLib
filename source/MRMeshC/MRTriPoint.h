#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRTriPointf
{
    /// ...
    float a;
    /// ...
    float b;
} MRTriPointf;

/// ...
MRMESHC_API MRTriPointf mrTriPointfFromTriangle( const MRVector3f* p, const MRVector3f* v0, const MRVector3f* v1, const MRVector3f* v2 );

/// ...
MRMESHC_API MRVector3f mrTriPointfInterpolate( const MRTriPointf* tp, const MRVector3f* v0, const MRVector3f* v1, const MRVector3f* v2 );

MR_EXTERN_C_END
