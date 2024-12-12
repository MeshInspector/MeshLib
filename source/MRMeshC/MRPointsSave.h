#pragma once

#include "MRMeshFwd.h"
#include "MRSaveSettings.h"

MR_EXTERN_C_BEGIN

/// detects the format from file extension and save points to it
MRMESHC_API void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc, const char* file, const MRSaveSettings* settings, MRString** errorString );

MR_EXTERN_C_END
