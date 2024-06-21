#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRMESHC_CLASS MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

MRMESHC_API MRMatrix3f mrMatrix3fIdentity();

#ifdef __cplusplus
}
#endif
