#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"

namespace MR
{

namespace FloatGridComponents
{

// finds separated by iso-value components in grid space (0 voxel id is minimum active voxel in grid)
MRMESH_API std::vector<VoxelBitSet> getAllComponents( const FloatGrid& grid, float isoValue = 0.0f );

}

}
#endif
