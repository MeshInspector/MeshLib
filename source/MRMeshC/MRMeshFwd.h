#pragma once

#ifdef __cplusplus
#include <cstddef>

extern "C"
{
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef _WIN32
#   ifdef MRMESHC_EXPORT
#       define MRMESHC_API __declspec( dllexport )
#   else
#       define MRMESHC_API __declspec( dllimport )
#   endif
#   define MRMESHC_CLASS
#else
#   define MRMESHC_API __attribute__( ( visibility( "default" ) ) )
#   define MRMESHC_CLASS __attribute__( ( visibility( "default" ) ) )
#endif

typedef struct MRString MRString;

typedef int MRVertId;
typedef int MRFaceId;

typedef MRVertId MRThreeVertIds[3];

typedef struct MRBitSet MRBitSet;
typedef MRBitSet MRFaceBitSet;

typedef struct MRVector3f MRVector3f;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;

typedef bool (*MRProgressCallback)( float );

#ifdef __cplusplus
}
#endif
