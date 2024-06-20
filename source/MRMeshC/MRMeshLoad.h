#pragma once

#include "MRMesh.h"
#include "MRString.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

#ifdef __cplusplus
}
#endif
