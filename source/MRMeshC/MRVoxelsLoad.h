#pragma once
#include "MRVoxelsFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

typedef struct MRVdbVolumes MRVdbVolumes;

/// gets the volumes' value at index
MRMESHC_API const MRVdbVolume mrVdbVolumesGet( const MRVdbVolumes* volumes, size_t index );

/// gets the volumes' size
MRMESHC_API size_t mrVdbVolumesSize( const MRVdbVolumes* volumes );

/// deallocates the VdbVolumes object
MRMESHC_API void mrVdbVolumesFree( MRVdbVolumes* volumes );

/// Detects the format from file extension and loads voxels from it
MRMESHC_API MRVdbVolumes* mrVoxelsLoadFromAnySupportedFormat( const char* file, MRProgressCallback cb, MRString** errorStr );

MR_EXTERN_C_END
