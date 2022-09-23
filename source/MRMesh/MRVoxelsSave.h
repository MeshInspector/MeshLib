#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include <tl/expected.hpp>
#include <filesystem>
#include "MRProgressCallback.h"

namespace MR
{

namespace VoxelsSave
{

/// \addtogroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

MRMESH_API tl::expected<void, std::string> saveRAW( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                    ProgressCallback callback = {} );

/// \}

}

}
#endif
