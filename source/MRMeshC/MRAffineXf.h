#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRMESHC_CLASS MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

MRMESHC_API MRAffineXf3f mrAffineXf3fNew();

#ifdef __cplusplus
}
#endif
