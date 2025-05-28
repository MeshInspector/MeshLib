#pragma once

#include "MRVoxelsFwd.h"
#ifndef MRVOXELS_NO_DICOM
#include "MRVoxelsVolume.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRLoadedObjects.h"

#include <filesystem>
#include <optional>

namespace MR
{

namespace VoxelsLoad
{


enum class DicomStatusEnum : int
{
    Ok = 0,         // valid DICOM and we can open it
    Invalid,        // not a valid DICOM
    Unsupported,    // a valid DICOM, but we do not support it (e.g. some MediaStorages)
};

struct DicomStatus
{
    // implicit by design
    DicomStatus( DicomStatusEnum st, std::string_view rs = "" ):
        status( st ),
        reason( rs )
    {}

    explicit operator bool() const
    { return status == DicomStatusEnum::Ok; }

    bool operator==( DicomStatusEnum s ) const
    { return status == s; }

    DicomStatusEnum status = DicomStatusEnum::Invalid;
    std::string reason;     // if status is Unsupported, specify reason why
};

/// check if file is a valid DICOM dataset file
/// \param seriesUid - if set, the extracted series instance UID is copied to the variable
MRVOXELS_API DicomStatus isDicomFile( const std::filesystem::path& path, std::string* seriesUid = nullptr );

/// check if given folder contains at least one DICOM file
MRVOXELS_API bool isDicomFolder( const std::filesystem::path& dirPath );

/// returns all the dicom folders in \p path, searching recursively
MRVOXELS_API std::vector<std::filesystem::path> findDicomFoldersRecursively( const std::filesystem::path& path );

template <typename T>
struct DicomVolumeT
{
    T vol;
    std::string name;
    AffineXf3f xf;
};

/// Loads full volume from single DICOM file (not a slice file) as SimpleVolumeMinMax
MRVOXELS_API Expected<DicomVolume> loadDicomFile( const std::filesystem::path& file, const ProgressCallback& cb = {} );

/// Loads full volume from single DICOM file (not a slice file) as VdbVolume
MRVOXELS_API Expected<DicomVolumeAsVdb> loadDicomFileAsVdb( const std::filesystem::path& file, const ProgressCallback& cb = {} );

/// Loads one volume from DICOM files located in given folder as SimpleVolumeMinMax
MRVOXELS_API Expected<DicomVolume> loadDicomFolder( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb = {} );

/// Loads one volume from DICOM files located in given folder as VdbVolume
MRVOXELS_API Expected<DicomVolumeAsVdb> loadDicomFolderAsVdb( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb = {} );

/// Loads all volumes from DICOM files located in given folder as a number of SimpleVolumeMinMax
MRVOXELS_API std::vector<Expected<DicomVolume>> loadDicomsFolder( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb = {} );

/// Loads all volumes from DICOM files located in given folder as a number of VdbVolume
MRVOXELS_API std::vector<Expected<DicomVolumeAsVdb>> loadDicomsFolderAsVdb( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRVOXELS_API std::vector<Expected<DicomVolumeAsVdb>> loadDicomsFolderTreeAsVdb( const std::filesystem::path& path,
                                                                     unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// converts DicomVolumeAsVdb in ObjectVoxels
MRVOXELS_API Expected<std::shared_ptr<ObjectVoxels>> createObjectVoxels( const DicomVolumeAsVdb & dcm, const ProgressCallback & cb = {} );

/// Loads 3D volumetric data from dicom-files in given folder, and converts them into an ObjectVoxels
MRVOXELS_API Expected<LoadedObjects> makeObjectVoxelsFromDicomFolder( const std::filesystem::path& folder, const ProgressCallback& callback = {} );

} // namespace VoxelsLoad

namespace VoxelsSave
{

/// Save voxels object to a single 3d DICOM file
MRVOXELS_API Expected<void> toDicom( const VdbVolume& vdbVolume, const std::filesystem::path& path, ProgressCallback cb = {} );
/// Saves object to a single 3d DICOM file. \p sourceScale specifies the true scale of the voxel data
/// which will be saved with "slope" and "intercept" parameters of the output dicom.
template <typename T>
MRVOXELS_API Expected<void> toDicom( const VoxelsVolume<Vector<T,VoxelId>>& volume, const std::filesystem::path& path, const std::optional<MinMaxf>& sourceScale = {}, const ProgressCallback& cb = {} );

extern template MRVOXELS_API Expected<void> toDicom( const VoxelsVolume<Vector<std::uint16_t, VoxelId>>& volume, const std::filesystem::path& path, const std::optional<MinMaxf>& sourceScale, const ProgressCallback& cb );

} // namespace VoxelsSave

} // namespace MR
#endif
