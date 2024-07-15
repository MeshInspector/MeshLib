#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

typedef struct MRVectorAffineXf3f MRVectorAffineXf3f;

MRMESHC_API const MRAffineXf3f* mrVectorAffineXf3fData( const MRVectorAffineXf3f* vec );

MRMESHC_API size_t mrVectorAffineXf3fSize( const MRVectorAffineXf3f* vec );

MRMESHC_API void meVectorAffineXf3fFree( MRVectorAffineXf3f* vec );

MR_EXTERN_C_END
