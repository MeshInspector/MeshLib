#pragma once
#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRObject.h"
#include "MRSimpleVolume.h"
#include "MRIOFilters.h"
#include <filesystem>
#include <tl/expected.hpp>

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

#if !defined(MRMESH_NO_DICOM)
struct LoadDCMResult
{
    VdbVolume vdbVolume;
    std::string name;
};

/// Loads data from DICOM file(s) to SimpleVolume
/// SimpleVolume dimensions: x,y equals to x,y dimensions of DICOM picture,
///                          z - number of pictures loaded
/// Files in folder are sorted by names
MRMESH_API tl::expected<LoadDCMResult, std::string> loadDCMFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRMESH_API std::vector<tl::expected<LoadDCMResult, std::string>> loadDCMFolderTree( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Load single DCM file as Object Voxels
MRMESH_API tl::expected<LoadDCMResult, std::string> loadDCMFile( const std::filesystem::path& path, const ProgressCallback& cb = {} );
#endif // MRMESH_NO_DICOM

struct RawParameters
{
    Vector3i dimensions;
    Vector3f voxelSize;
    bool gridLevelSet{ false }; ///< OpenVDB GridClass set as GRID_LEVEL_SET (need to set right surface normals direction)
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
        Unknown,
        Count
    } scalarType{ ScalarType::Float32 };
};
/// Load raw voxels file with provided parameters
MRMESH_API tl::expected<VdbVolume, std::string> loadRaw( const std::filesystem::path& path, const RawParameters& params,
                                                         const ProgressCallback& cb = {} );

/// Load raw voxels file, parsing parameters from name 
MRMESH_API tl::expected<VdbVolume, std::string> loadRaw( const std::filesystem::path& path,
                                                         const ProgressCallback& cb = {} );

/// Load raw voxels file, parsing parameters from name 
MRMESH_API tl::expected<VdbVolume, std::string> fromVdb( const std::filesystem::path& path,
                                                         const ProgressCallback& cb = {} );

/// Detects the format from file extension and loads voxels from it
MRMESH_API tl::expected<VdbVolume, std::string> fromAnySupportedFormat( const std::filesystem::path& path,
                                                                        const ProgressCallback& cb = {} );

/// \}

// Determines iso-surface orientation
enum class GridType
{
    // consider values less than iso as outer area
    DenseGrid = 0,
    // consider values less than iso as inner area
    LevelSet = 1
};

#ifndef MRMESH_NO_TIFF
struct LoadingTiffSettings
{
    std::filesystem::path dir;
    Vector3f voxelSize = { 1.0f, 1.0f, 1.0f };
    GridType gridType = GridType::DenseGrid;
    ProgressCallback cb = {};
};
/// Load voxels from a set of TIFF files
MRMESH_API tl::expected<VdbVolume, std::string> loadTiffDir( const LoadingTiffSettings& settings );
#endif // MRMESH_NO_TIFF
}

}
#endif // !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXELS )
