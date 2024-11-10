#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// three-dimensional vector
typedef struct MRVector3f
{
    float x;
    float y;
    float z;
} MRVector3f;

/// (a, a, a)
MRMESHC_API MRVector3f mrVector3fDiagonal( float a );

/// (1, 0, 0)
MRMESHC_API MRVector3f mrVector3fPlusX( void );

/// (0, 1, 0)
MRMESHC_API MRVector3f mrVector3fPlusY( void );

/// (0, 0, 1)
MRMESHC_API MRVector3f mrVector3fPlusZ( void );

/// adds two vectors
MRMESHC_API MRVector3f mrVector3fAdd( const MRVector3f* a, const MRVector3f* b );

MRMESHC_API MRVector3f mrVector3fSub( const MRVector3f* a, const MRVector3f* b );

/// multiplies a vector by a scalar value
MRMESHC_API MRVector3f mrVector3fMulScalar( const MRVector3f* a, float b );

/// squared length of the vector
MRMESHC_API float mrVector3fLengthSq( const MRVector3f* v );

/// length of the vector
MRMESHC_API float mrVector3fLength( const MRVector3f* v );

/// a set of 3 vectors; useful for representing a face via its vertex coordinates
typedef MRVector3f MRTriangle3f[3];

MR_EXTERN_C_END
