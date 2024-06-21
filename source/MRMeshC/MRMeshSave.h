#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr );

#ifdef __cplusplus
}
#endif
