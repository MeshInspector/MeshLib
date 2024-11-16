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

MRVOXELS_API VdbVolume voxelFilter( const VdbVolume& volume, VoxelFilterType type, int width );

}