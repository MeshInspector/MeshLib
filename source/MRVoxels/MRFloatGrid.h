#pragma once
#include "MRVoxelsFwd.h"

// this is a lightweight header unlike MRVDBFloatGrid.h

#include "MRMesh/MRProgressCallback.h"

namespace MR
{

/**
 * \defgroup BasicStructuresGroup Basic Structures
 * \brief This chapter represents documentation about basic structures elements
 * \{
 */

/// returns the amount of heap memory occupied by grid
[[nodiscard]] MRVOXELS_API size_t heapBytes( const FloatGrid& grid );

/// resample this grid to fit voxelScale
MRVOXELS_API FloatGrid resampled( const FloatGrid& grid, float voxelScale, ProgressCallback cb = {} );

/// resample this grid to fit voxelScale
MRVOXELS_API FloatGrid resampled( const FloatGrid& grid, const Vector3f& voxelScale, ProgressCallback cb = {} );

/// returns cropped grid
MRVOXELS_API FloatGrid cropped( const FloatGrid& grid, const Box3i& box, ProgressCallback cb = {} );

/// returns the value at given voxel
[[nodiscard]] MRVOXELS_API float getValue( const FloatGrid & grid, const Vector3i & p );

/// sets given region voxels value
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRVOXELS_API void setValue( FloatGrid & grid, const VoxelBitSet& region, float value );

/// sets type of this grid as LEVEL SET (for normal flipping)
MRVOXELS_API void setLevelSetType( FloatGrid & grid );

// union operation on volumetric representation of two meshes
MRVOXELS_API FloatGrid operator += ( FloatGrid & a, const FloatGrid& b );

// difference operation on volumetric representation of two meshes
MRVOXELS_API FloatGrid operator -= ( FloatGrid & a, const FloatGrid& b );

// intersection operation on volumetric representation of two meshes
MRVOXELS_API FloatGrid operator *= ( FloatGrid & a, const FloatGrid& b );

/// \}

}
