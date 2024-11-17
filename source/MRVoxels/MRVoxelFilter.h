#pragma once

#include "MRVoxelsFwd.h"


namespace MR
{

enum class VoxelFilterType : int
{
    Median,
    Mean,
    Gaussian
};

/// Performs voxels filtering.
/// @param type Type of fitler
/// @param width Width of the filtering window, must be an odd number greater or equal to 1.
MRVOXELS_API VdbVolume voxelFilter( const VdbVolume& volume, VoxelFilterType type, int width );

}