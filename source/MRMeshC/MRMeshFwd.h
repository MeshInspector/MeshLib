#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef _WIN32
#   ifdef MRMESHC_EXPORT
#       define MRMESHC_API __declspec( dllexport )
#   else
#       define MRMESHC_API __declspec( dllimport )
#   endif
#else
#   define MRMESHC_API __attribute__( ( visibility( "default" ) ) )
#endif

#ifdef __cplusplus
#define MR_EXTERN_C_BEGIN extern "C" {
#define MR_EXTERN_C_END }
#else
#define MR_EXTERN_C_BEGIN
#define MR_EXTERN_C_END
#endif

MR_EXTERN_C_BEGIN

typedef struct MRString MRString;

typedef struct MRBitSet MRBitSet;
typedef struct MREdgeBitSet MREdgeBitSet;
typedef struct MRUndirectedEdgeBitSet MRUndirectedEdgeBitSet;
typedef struct MRFaceBitSet MRFaceBitSet;
typedef struct MRVertBitSet MRVertBitSet;

typedef struct MRVector3f MRVector3f;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;

typedef struct MRPointCloud MRPointCloud;

typedef bool (*MRProgressCallback)( float );

MR_EXTERN_C_END
