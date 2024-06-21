#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

#ifdef __cplusplus
}
#endif
