#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// detects the format from file extension and save points to it
MRMESHC_API void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc, const char* file, MRString** errorString );

MR_EXTERN_C_END
