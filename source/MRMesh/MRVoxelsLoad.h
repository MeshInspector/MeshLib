#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRObject.h"
#include "MRSimpleVolume.h"
#include "MRIOFilters.h"
#include <filesystem>

namespace MR
{

namespace VoxelsLoad
{

/// \defgroup VoxelsLoadGroup Voxels Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// Sort files in given vector by names (respect numbers in it)
/// usually needed for scans
MRMESH_API void sortFilesByName( std::vector<std::filesystem::path>& scans );

/// Loads data from DICOM file(s) to SimpleVolume
/// SimpleVolume dimensions: x,y equals to x,y dimensions of DICOM picture,
///                          z - number of pictures loaded
/// Files in folder are sorted by names
MRMESH_API std::shared_ptr<ObjectVoxels> loadDCMFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4,
                                                        const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRMESH_API std::vector<std::shared_ptr<ObjectVoxels>> loadDCMFolderTree( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4,
                                                        const ProgressCallback& cb = {} );

/// Load single DCM file as Object Voxels
MRMESH_API std::shared_ptr<ObjectVoxels> loadDCMFile( const std::filesystem::path& path,
                                                      const ProgressCallback& cb = {} );

struct RawParameters
{
    Vector3i dimensions;
    Vector3f voxelSize;
    enum class ScalarType
    {
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        UInt64,
        Int64,
        Float32,
        Float64,
        Count
    } scalarType{ ScalarType::Float32 };
};
/// Load raw voxels file with provided parameters
MRMESH_API tl::expected<SimpleVolume, std::string> loadRaw( const std::filesystem::path& path, const RawParameters& params,
                                                      const ProgressCallback& cb = {} );

/// Load raw voxels file, parsing parameters from name 
MRMESH_API tl::expected<SimpleVolume, std::string> loadRaw( const std::filesystem::path& path,
                                                      const ProgressCallback& cb = {} );

/// \}

}

}
#endif
