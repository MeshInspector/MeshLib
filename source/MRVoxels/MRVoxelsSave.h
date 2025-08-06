#pragma once
#include "MRVoxelsFwd.h"
#include "MRVoxelPath.h"
#include "MRVoxelsVolume.h"

#include "MRMesh/MRIOFormatsRegistry.h"

#include <filesystem>

namespace MR
{

namespace VoxelsSave
{

/// \addtogroup IOGroup
/// \{

/// Save raw voxels file, writing parameters in file name
MRVOXELS_API Expected<void> toRawAutoname( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                       ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toRawAutoname( const SimpleVolume& simpleVolume, const std::filesystem::path& file,
                                       ProgressCallback callback = {} );
MRVOXELS_API Expected<void> gridToRawAutoname( const FloatGrid& grid, const Vector3i& dims, const std::filesystem::path& file,
    ProgressCallback callback = {} );

/// Save voxels in raw format with each value as 32-bit float in given binary stream
MRVOXELS_API Expected<void> gridToRawFloat( const FloatGrid& grid, const Vector3i& dims, std::ostream& out, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toRawFloat( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toRawFloat( const SimpleVolume& simpleVolume, std::ostream & out, ProgressCallback callback = {} );

/// Save voxels in Gav-format in given destination
MRVOXELS_API Expected<void> toGav( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toGav( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toGav( const SimpleVolumeMinMax& simpleVolumeMinMax, const std::filesystem::path& file, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toGav( const SimpleVolumeMinMax& simpleVolumeMinMax, std::ostream & out, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toGav( const SimpleVolume& simpleVolume, const std::filesystem::path& file, ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toGav( const SimpleVolume& simpleVolume, std::ostream & out, ProgressCallback callback = {} );

/// Save voxels file in OpenVDB format
MRVOXELS_API Expected<void> gridToVdb( const FloatGrid& grid, const std::filesystem::path& file,
                               ProgressCallback callback = {} );
MRVOXELS_API Expected<void> gridToVdb( const FloatGrid& grid, std::ostream& out,
                               ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toVdb( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                               ProgressCallback callback = {} );

/// Saves voxels in a file, detecting the format from file extension
MRVOXELS_API Expected<void> gridToAnySupportedFormat( const FloatGrid& grid, const Vector3i& dims, const std::filesystem::path& file,
                                              ProgressCallback callback = {} );
MRVOXELS_API Expected<void> toAnySupportedFormat( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                              ProgressCallback callback = {} );

/// save the slice by the active plane through the sliceNumber to an image file
MRVOXELS_API Expected<void> saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlane& slicePlain, int sliceNumber, ProgressCallback callback = {} );

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
MRVOXELS_API Expected<void> saveAllSlicesToImage( const VdbVolume& vdbVolume, const SavingSettings& settings );

/// \}

#ifndef MR_PARSING_FOR_ANY_BINDINGS
using VoxelsSaver = Expected<void>( * )( const VdbVolume&, const std::filesystem::path&, ProgressCallback );

MR_FORMAT_REGISTRY_EXTERNAL_DECL( MRVOXELS_API, VoxelsSaver )
#endif

} // namespace VoxelsSave

MRVOXELS_API Expected<void> saveObjectVoxelsToFile( const Object& object, const std::filesystem::path& path,
                                                    const ProgressCallback& callback = {} );

} // namespace MR
