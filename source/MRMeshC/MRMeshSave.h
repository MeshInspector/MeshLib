#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr );

MR_EXTERN_C_END
