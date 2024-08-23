#pragma once
#include "MRVoxelsFwd.h"
#include "MRMesh/MRIOFilters.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRVoxelPath.h"
#include "MRVoxelsVolume.h"
#include <filesystem>

namespace MR
{

namespace VoxelsSave
{

/// \addtogroup IOGroup
/// \{

MRVOXELS_API extern const IOFilters Filters;

/// Save raw voxels file, writing parameters in file name
MRVOXELS_API VoidOrErrStr toRawAutoname( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                       ProgressCallback callback = {} );

/// Save voxels in raw format with each value as 32-bit float in given binary stream
MRVOXELS_API VoidOrErrStr toRawFloat( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback = {} );

/// Save voxels in Gav-format in given file
MRVOXELS_API VoidOrErrStr toGav( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback = {} );
/// Save voxels in Gav-format in given binary stream
MRVOXELS_API VoidOrErrStr toGav( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback = {} );

/// Save voxels file in OpenVDB format
MRVOXELS_API VoidOrErrStr toVdb( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                               ProgressCallback callback = {} );

/// Saves voxels in a file, detecting the format from file extension
MRVOXELS_API VoidOrErrStr toAnySupportedFormat( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                              ProgressCallback callback = {} );

/// save the slice by the active plane through the sliceNumber to an image file
MRVOXELS_API VoidOrErrStr saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlane& slicePlain, int sliceNumber, ProgressCallback callback = {} );

// stores together all data for save voxel object as a group of images
struct SavingSettings
{
    // path to directory where you want to save images
    std::filesystem::path path;
    // format for file names, you should specify a placeholder for number and extension, eg "slice_{0:0{1}}.tif"
    std::string format = "slice_{0:0{1}}.tif";
    // Plane which the object is sliced by. XY, XZ, or YZ
    SlicePlane slicePlane;
    // Callback reporting progress
    ProgressCallback cb = {};
};

/// save all slices by the active plane through all voxel planes along the active axis to an image file
MRVOXELS_API VoidOrErrStr saveAllSlicesToImage( const VdbVolume& vdbVolume, const SavingSettings& settings );

/// \}

} // namespace VoxelsSave

MRVOXELS_API Expected<void> saveObjectVoxelsToFile( const Object& object, const std::filesystem::path& path,
                                                    ProgressCallback callback = {} );

} // namespace MR
