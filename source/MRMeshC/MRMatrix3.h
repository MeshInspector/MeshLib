#pragma once

#include "MRVector3.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

MRMatrix3f mrMatrix3fIdentity();

#ifdef __cplusplus
}
#endif
