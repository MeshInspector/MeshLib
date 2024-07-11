#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// detects the format from file extension and loads points from it
MRMESHC_API MRPointCloud* mrPointsLoadFromAnySupportedFormat( const char* filename, MRString** errorString );

MR_EXTERN_C_END
