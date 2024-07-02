#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRVector3f
{
    float x;
    float y;
    float z;
} MRVector3f;

MRMESHC_API MRVector3f mrVector3fDiagonal( float a );

MRMESHC_API MRVector3f mrVector3fPlusX( void );

MRMESHC_API MRVector3f mrVector3fPlusY( void );

MRMESHC_API MRVector3f mrVector3fPlusZ( void );

MRMESHC_API MRVector3f mrVector3fAdd( const MRVector3f* a, const MRVector3f* b );

MRMESHC_API MRVector3f mrVector3fMulScalar( const MRVector3f* a, float b );

typedef MRVector3f MRTriangle3f[3];

MR_EXTERN_C_END
