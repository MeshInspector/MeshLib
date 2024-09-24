#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRVector3.h"

#include <compare>
#include <filesystem>

namespace MR
{

/// slice information
/// \sa SliceInfo
struct SliceInfoBase
{
    /// instance number
    int instanceNum = 0;
    /// layer height
    double z = 0;
    /// file index
    int fileNum = 0;

    auto operator <=>( const SliceInfoBase & ) const = default;
};

/// slice information
/// these fields will be ignored in sorting
/// \sa SliceInfoBase
struct SliceInfo : SliceInfoBase
{
    /// image position
    Vector3d imagePos;
};

/// Sort scan files in given vector by given slice information
MRVOXELS_API void sortScansByOrder( std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder );

/// Read layer heights from given scan file names
MRVOXELS_API void putScanFileNameInZ( const std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder );

/// Sort scan files in given vector by names (respect numbers in it)
MRVOXELS_API void sortScanFilesByName( std::vector<std::filesystem::path>& scans );

} // namespace MR
