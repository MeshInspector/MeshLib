#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"

MR_EXTERN_C_BEGIN

typedef struct MRMESHC_CLASS MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

MRMESHC_API MRAffineXf3f mrAffineXf3fNew();

MR_EXTERN_C_END
