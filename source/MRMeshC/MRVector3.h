#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRMESHC_CLASS MRVector3f
{
    float x;
    float y;
    float z;
} MRVector3f;

MRMESHC_API MRVector3f mrVector3fDiagonal( float a );

MRMESHC_API MRVector3f mrVector3fPlusX();

MRMESHC_API MRVector3f mrVector3fPlusY();

MRMESHC_API MRVector3f mrVector3fPlusZ();

MRMESHC_API MRVector3f mrVector3fAdd( const MRVector3f* a, const MRVector3f* b );

MRMESHC_API MRVector3f mrVector3fMulScalar( const MRVector3f* a, float b );

MR_EXTERN_C_END
