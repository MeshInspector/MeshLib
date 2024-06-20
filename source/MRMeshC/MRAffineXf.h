#pragma once

#include "MRMatrix3.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

MRAffineXf3f mrAffineXf3fNew();

#ifdef __cplusplus
}
#endif
