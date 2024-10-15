#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

/// arbitrary row-major 3x3 matrix
typedef struct MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

/// initializes an identity matrix
MRMESHC_API MRMatrix3f mrMatrix3fIdentity( void );

/// creates a matrix representing rotation around given axis on given angle
MRMESHC_API MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis, float angle );

/// creates a matrix representing rotation that after application to (from) makes (to) vector
MRMESHC_API MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from, const MRVector3f* to );

/// multiplies two matrices
MRMESHC_API MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a, const MRMatrix3f* b );

MRMESHC_API MRMatrix3f mrMatrix3fAdd( const MRMatrix3f* a, const MRMatrix3f* b );

MRMESHC_API MRMatrix3f mrMatrix3fSub( const MRMatrix3f* a, const MRMatrix3f* b );

MRMESHC_API MRVector3f mrMatrix3fMulVector( const MRMatrix3f* a, const MRVector3f* b );

MRMESHC_API bool mrMatrix3fEqual( const MRMatrix3f* a, const MRMatrix3f* b );

MR_EXTERN_C_END
