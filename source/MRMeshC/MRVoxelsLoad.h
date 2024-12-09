#pragma once
#include "MRVoxelsFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

MR_VECTOR_LIKE_DECL( VdbVolumes, VdbVolume )

/// Detects the format from file extension and loads voxels from it
MRVdbVolumes* mrVoxelsLoadFromAnySupportedFormat( const char* file, MRProgressCallback cb, MRString** errorStr );

MR_EXTERN_C_END
