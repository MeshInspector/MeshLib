#pragma once
#include "MRVoxelsFwd.h"
#include "MRVoxelsVolume.h"

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRObject.h"

#include <filesystem>

namespace MR
{

namespace VoxelsLoad
{

/// \defgroup VoxelsLoadGroup Voxels Load
/// \ingroup IOGroup
/// \{

/// Sort files in given vector by names (respect numbers in it)
/// usually needed for scans
MRVOXELS_API void sortFilesByName( std::vector<std::filesystem::path>& scans );

#ifndef MRVOXELS_NO_DICOM
struct DicomVolume
{
    SimpleVolumeMinMax vol;
    std::string name;
    AffineXf3f xf;
};

struct LoadDCMResult
{
    VdbVolume vdbVolume;
    std::string name;
    AffineXf3f xf;
};

/// Loads 3D all volumetric data from DICOM files in a folder
MRVOXELS_API std::vector<Expected<LoadDCMResult>> loadDCMsFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );
/// Loads 3D first volumetric data from DICOM files in a folder
MRVOXELS_API Expected<LoadDCMResult> loadDCMFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads 3D all volumetric data from DICOM files in a folder
MRVOXELS_API std::vector<Expected<DicomVolume>> loadDicomsFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );
/// Loads 3D first volumetric data from DICOM files in a folder
MRVOXELS_API Expected<DicomVolume> loadDicomFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRVOXELS_API std::vector<Expected<LoadDCMResult>> loadDCMFolderTree( const std::filesystem::path& path,
                                                        unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads 3D volumetric data from a single DICOM file
MRVOXELS_API Expected<DicomVolume> loadDicomFile( const std::filesystem::path& path, const ProgressCallback& cb = {} );
#endif // MRVOXELS_NO_DICOM

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
        Float32_4, ///< the last value from float[4]
        Unknown,
        Count
    } scalarType{ ScalarType::Float32 };
};
/// Load raw voxels from file with provided parameters
MRVOXELS_API Expected<VdbVolume> fromRaw( const std::filesystem::path& file, const RawParameters& params,
                                                         const ProgressCallback& cb = {} );
/// Load raw voxels from stream with provided parameters;
/// important on Windows: in stream must be open in binary mode
MRVOXELS_API Expected<VdbVolume> fromRaw( std::istream& in, const RawParameters& params,
                                                         const ProgressCallback& cb = {} );

/// finds raw voxels file and its encoding parameters
/// \param file on input: file name probably without suffix with parameters
///             on output: if success existing file name
MRVOXELS_API Expected<RawParameters> findRawParameters( std::filesystem::path& file );

/// Load raw voxels file, parsing parameters from name 
MRVOXELS_API Expected<VdbVolume> fromRaw( const std::filesystem::path& file,
                                                         const ProgressCallback& cb = {} );

/// Load all voxel volumes from OpenVDB file
MRVOXELS_API Expected<std::vector<VdbVolume>> fromVdb( const std::filesystem::path& file,
                                                         const ProgressCallback& cb = {} );

/// Load voxel from Gav-file with micro CT reconstruction
MRVOXELS_API Expected<VdbVolume> fromGav( const std::filesystem::path& file, const ProgressCallback& cb = {} );
/// Load voxel from Gav-stream with micro CT reconstruction
MRVOXELS_API Expected<VdbVolume> fromGav( std::istream& in, const ProgressCallback& cb = {} );


/// Detects the format from file extension and loads voxels from it
MRVOXELS_API Expected<std::vector<VdbVolume>> fromAnySupportedFormat( const std::filesystem::path& file,
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

#ifndef MRVOXELS_NO_TIFF
struct LoadingTiffSettings
{
    std::filesystem::path dir;
    Vector3f voxelSize = { 1.0f, 1.0f, 1.0f };
    GridType gridType = GridType::DenseGrid;
    ProgressCallback cb = {};
};
/// Load voxels from a set of TIFF files
MRVOXELS_API Expected<VdbVolume> loadTiffDir( const LoadingTiffSettings& settings );
#endif // MRVOXELS_NO_TIFF

#ifndef MR_PARSING_FOR_PB11_BINDINGS
using VoxelsLoader = Expected<std::vector<VdbVolume>>( * )( const std::filesystem::path&, const ProgressCallback& );

MR_FORMAT_REGISTRY_EXTERNAL_DECL( MRVOXELS_API, VoxelsLoader )
#endif

}

/// loads voxels from given file in new object
MRVOXELS_API Expected<std::vector<std::shared_ptr<ObjectVoxels>>> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

MRVOXELS_API Expected<std::vector<std::shared_ptr<Object>>> makeObjectFromVoxelsFile( const std::filesystem::path& file, std::string* warnings = nullptr, ProgressCallback callback = {} );

}
