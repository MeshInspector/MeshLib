#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

typedef struct MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

MRMESHC_API MRMatrix3f mrMatrix3fIdentity();

MRMESHC_API MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis, float angle );

MRMESHC_API MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from, const MRVector3f* to );

MRMESHC_API MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a, const MRMatrix3f* b );

MR_EXTERN_C_END
