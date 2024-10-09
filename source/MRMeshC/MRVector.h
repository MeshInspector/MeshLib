#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "detail/Concat.h"

MR_EXTERN_C_BEGIN

#define MR_VECTOR_LIKE_DECL( ClassName, Type ) \
typedef struct MR_CONCAT( MR, ClassName )      \
{                                              \
    MR_CONCAT( MR, Type )* data;               \
    size_t size;                               \
} MR_CONCAT( MR, ClassName );                  \
MRMESHC_API void MR_CONCAT( MR_CONCAT( mr, ClassName ), Invalidate )( MR_CONCAT( MR, ClassName )* ); \
MRMESHC_API void MR_CONCAT( MR_CONCAT( mr, ClassName ), Free )( MR_CONCAT( MR, ClassName )* );

#define MR_VECTOR_DECL( Type ) MR_VECTOR_LIKE_DECL( MR_CONCAT( Vector, Type ), Type )

MR_VECTOR_DECL( AffineXf3f )
MR_VECTOR_DECL( Vector3f )

MR_EXTERN_C_END
