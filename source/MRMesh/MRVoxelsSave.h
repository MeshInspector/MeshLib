#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include <tl/expected.hpp>
#include <filesystem>
#include "MRProgressCallback.h"
#include "MRVoxelPath.h"
#include "MRSimpleVolume.h"

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

// stores together all data for save voxel object as a group of images
struct SavingSettings
{
    // path to directory where you want to save images
    std::filesystem::path path;
    // format for file names, you should specify a placeholder for number and extension, eg "slice_{0:0{1}}.tif"
    fmt::format_string<int, int> format = "slice_{0:0{1}}.tif";
    // voxel data
    VdbVolume vdbVolume;
    // Plain which the object is sliced by. XY, XZ, or YZ
    SlicePlain slicePlain;
    // Callback reporting progress
    ProgressCallback cb = {};
};

/// save all slices by the active plane through all voxel planes along the active axis to an image file
MRMESH_API tl::expected<void, std::string> saveAllSlicesToImage( const SavingSettings& settings );

/// \}

}

}
#endif
