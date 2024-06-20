#pragma once


#ifdef __cplusplus
#include <cstddef>

extern "C"
{
#else
#include <stdbool.h>
#include <stddef.h>
#endif

typedef struct MRString MRString;

typedef int MRVertId;
typedef int MRFaceId;

typedef MRVertId MRThreeVertIds[3];

typedef struct MRBitSet MRBitSet;

typedef struct MRVector3f MRVector3f;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;

typedef bool (*MRProgressCallback)( float );

#ifdef __cplusplus
}
#endif
