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
#   ifdef MRMeshC_EXPORTS
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
typedef struct MRVoxelBitSet MRVoxelBitSet;

typedef struct MRBox3f MRBox3f;
typedef struct MRBox3i MRBox3i;
typedef struct MRVector3f MRVector3f;
typedef struct MRAffineXf3f MRAffineXf3f;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;
typedef struct MRMeshPart MRMeshPart;

typedef struct MRPointCloud MRPointCloud;
typedef struct MRPointsLoadSettings MRPointsLoadSettings;

typedef struct MRSaveSettings MRSaveSettings;

typedef bool (*MRProgressCallback)( float );

#ifdef __cplusplus
#define MRSTD std::
#else
#define MRSTD
#endif

#ifdef __APPLE__
typedef MRSTD ptrdiff_t Int64;
typedef MRSTD size_t Uint64;
#ifdef __cplusplus
static_assert(sizeof(Int64) == 8);
static_assert(sizeof(Uint64) == 8);
#endif
#else //!__APPLE__
typedef MRSTD int64_t Int64;
typedef MRSTD uint64_t Uint64;
#endif

MR_EXTERN_C_END
