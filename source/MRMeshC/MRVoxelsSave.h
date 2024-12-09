#pragma once
#include "MRVoxelsFwd.h"

MR_EXTERN_C_BEGIN

/// Saves voxels in a file, detecting the format from file extension
MRMESHC_API void mrVoxelsSaveToAnySupportedFormat( const MRVdbVolume* volume, const char* file, MRProgressCallback cb, MRString** errorStr );

MR_EXTERN_C_END
