#pragma once

#include "MRMeshFwd.h"
#include "MRSaveSettings.h"

MR_EXTERN_C_BEGIN

/// detects the format from file extension and saves mesh to it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, const MRSaveSettings* settings, MRString** errorStr );

MR_EXTERN_C_END
