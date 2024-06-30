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

MRMESHC_API MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b );

MRMESHC_API MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A );

MRMESHC_API MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a, const MRAffineXf3f* b );

MR_EXTERN_C_END
