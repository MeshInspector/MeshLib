#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

/// concat tokens
#define MR_VECTOR_CONCAT( a, b ) MR_VECTOR_CONCAT_( a, b )
#define MR_VECTOR_CONCAT_( a, b ) a ## b

/// To simplify access to C++ array containers (aka std::vector), we use std::span-like structs to store
/// a pointer to data and array length. There are two important notes about using these structs:
///  - if a function returns a pointer to a vector-like struct, you must call mrVector*Free function
///    to deallocate it manually
///  - if any operation resizes the underlying vector, the data pointer might be invalidated; it's better to call
///    mrVector*Invalidate to update the pointer after any operation with the vector
#define MR_VECTOR_LIKE_DECL_NONS( ClassName, Type )   \
typedef struct MR_VECTOR_CONCAT( MR, ClassName ) \
{                                                \
    Type* data;                                  \
    size_t size;                                 \
    void* reserved1;                             \
} MR_VECTOR_CONCAT( MR, ClassName );             \
MRMESHC_API void MR_VECTOR_CONCAT( MR_VECTOR_CONCAT( mr, ClassName ), Invalidate )( MR_VECTOR_CONCAT( MR, ClassName )* ); \
MRMESHC_API void MR_VECTOR_CONCAT( MR_VECTOR_CONCAT( mr, ClassName ), Free )( MR_VECTOR_CONCAT( MR, ClassName )* ); \
MRMESHC_API MR_VECTOR_CONCAT( MR, ClassName )* MR_VECTOR_CONCAT( MR_VECTOR_CONCAT( mr, ClassName ), New )(void);

#define MR_VECTOR_LIKE_DECL( ClassName, Type ) MR_VECTOR_LIKE_DECL_NONS( ClassName, MR_VECTOR_CONCAT( MR, Type ) )
#define MR_VECTOR_DECL( Type ) MR_VECTOR_LIKE_DECL_NONS( MR_VECTOR_CONCAT( Vector, Type ), MR_VECTOR_CONCAT( MR, Type ) )

MR_VECTOR_DECL( AffineXf3f )
MR_VECTOR_DECL( Vector3f )

MR_VECTOR_LIKE_DECL( FaceMap, FaceId )
MR_VECTOR_LIKE_DECL( WholeEdgeMap, EdgeId )
MR_VECTOR_LIKE_DECL( VertMap, VertId )

MR_VECTOR_LIKE_DECL_NONS( Scalars, float )

MR_EXTERN_C_END
