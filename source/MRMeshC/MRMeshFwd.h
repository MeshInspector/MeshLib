#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MRString MRString;

typedef int MRVertId;
typedef int MRFaceId;

typedef MRVertId MRThreeVertIds[3];

typedef struct MRBitSet MRBitSet;

typedef struct MRVector3f MRVector3f;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;

#ifdef __cplusplus
}
#endif
