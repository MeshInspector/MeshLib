#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

MR_EXTERN_C_END
