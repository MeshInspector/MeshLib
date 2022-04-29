#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"

namespace MR
{

namespace FloatGridComponents
{

/**
 * \defgroup ComponentsGroup Components
 * \brief This chapter represents documentation about components
 * \{
 */

/// finds separated by iso-value components in grid space (0 voxel id is minimum active voxel in grid)
/// \ingroup ComponentsGroup
MRMESH_API std::vector<VoxelBitSet> getAllComponents( const FloatGrid& grid, float isoValue = 0.0f );

}

}
#endif
