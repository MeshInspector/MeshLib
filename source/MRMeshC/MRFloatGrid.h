#pragma once
#include "MRVoxelsFwd.h"

MR_EXTERN_C_BEGIN

/// resample this grid with voxel size uniformly scaled by voxelScale
MRMESHC_API MRFloatGrid* mrFloatGridResampledUniformly( const MRFloatGrid* grid, float voxelScale, MRProgressCallback cb );

/// resample this grid with voxel size scaled by voxelScale in each dimension
MRMESHC_API MRFloatGrid* mrFloatGridResampled( const MRFloatGrid* grid, const MRVector3f* voxelScale, MRProgressCallback cb );

/// returns cropped grid
MRMESHC_API MRFloatGrid* mrFloatGridCropped( const MRFloatGrid* grid, const MRBox3i* box, MRProgressCallback cb );

/// returns the value at given voxel
MRMESHC_API float mrFloatGridGetValue( const MRFloatGrid* grid, const MRVector3i* p );

/// sets given voxel
MRMESHC_API void mrFloatGridSetValue( MRFloatGrid* grid, const MRVector3i* p, float value );

/// sets given region voxels value
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRMESHC_API void mrFloatGridSetValueForRegion( MRFloatGrid* grid, const MRVoxelBitSet* region, float value );

MR_EXTERN_C_END

