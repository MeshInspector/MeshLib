#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

/// concat tokens
#define MR_CONCAT( a, b ) MR_CONCAT_( a, b )
#define MR_CONCAT_( a, b ) a ## b

/// To simplify access to C++ array containers (aka std::vector), we use std::span-like structs to store
/// a pointer to data and array length. There are two important notes about using these structs:
///  - if a function returns a pointer to a vector-like struct, you must call mrVector*Free function
//     to deallocate it manually
///  - if any operation resizes the underlying vector, the data pointer might be invalidated; it's better to call
///    mrVector*Invalidate to update the pointer after any operation with the vector
#define MR_VECTOR_LIKE_DECL( ClassName, Type ) \
typedef struct MR_CONCAT( MR, ClassName )      \
{                                              \
    MR_CONCAT( MR, Type )* data;               \
    size_t size;                               \
    void* reserved1;                           \
} MR_CONCAT( MR, ClassName );                  \
MRMESHC_API void MR_CONCAT( MR_CONCAT( mr, ClassName ), Invalidate )( MR_CONCAT( MR, ClassName )* ); \
MRMESHC_API void MR_CONCAT( MR_CONCAT( mr, ClassName ), Free )( MR_CONCAT( MR, ClassName )* );

#define MR_VECTOR_DECL( Type ) MR_VECTOR_LIKE_DECL( MR_CONCAT( Vector, Type ), Type )

MR_VECTOR_DECL( AffineXf3f )
MR_VECTOR_DECL( Vector3f )

MR_VECTOR_LIKE_DECL( FaceMap, FaceId )
MR_VECTOR_LIKE_DECL( WholeEdgeMap, EdgeId )
MR_VECTOR_LIKE_DECL( VertMap, VertId )

#undef MR_CONCAT_
#undef MR_CONCAT

MR_EXTERN_C_END
