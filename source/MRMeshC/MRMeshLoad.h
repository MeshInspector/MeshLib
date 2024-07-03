#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// detects the format from file extension and loads mesh from it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

MR_EXTERN_C_END
