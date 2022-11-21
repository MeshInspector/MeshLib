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

/// Save raw voxels file, writing parameters in name
MRMESH_API tl::expected<void, std::string> saveRaw( const std::filesystem::path& path, const VdbVolume& vdbVolume,
                                                    ProgressCallback callback = {} );

/// save the slice by the active plane through the sliceNumber to an image file
MRMESH_API tl::expected<void, std::string> saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlain& slicePlain, int sliceNumber, ProgressCallback callback = {} );
/// save all slices by the active plane through all voxel planes along the active axis to an image file
MRMESH_API tl::expected<void, std::string> saveAllSlicesToImage( const std::filesystem::path& path, const std::string& extension, const VdbVolume& vdbVolume, const SlicePlain& slicePlain, ProgressCallback callback = {} );

/// \}

}

}
#endif
