#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include <tl/expected.hpp>
#include <filesystem>
#include "MRProgressCallback.h"
#include "MRVoxelPath.h"

namespace MR
{

namespace VoxelsSave
{

/// \addtogroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

MRMESH_API tl::expected<void, std::string> saveRAW( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                    ProgressCallback callback = {} );

//save the slice by the active plane through the sliceNumber to an image file
MRMESH_API tl::expected<void, std::string> saveSliceToImage( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                             SlicePlain slicePlain, int sliceNumber, float min, float max, ProgressCallback callback = {} );
//save all slices by the active plane through all voxel planes along the active axis to an image file
MRMESH_API tl::expected<void, std::string> saveAllSlicesToImage( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                             SlicePlain slicePlain, float min, float max, ProgressCallback callback = {} );

/// \}

}

}
#endif
