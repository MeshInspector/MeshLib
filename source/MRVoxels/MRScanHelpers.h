#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRVector3.h"

#include <compare>
#include <filesystem>

namespace MR
{

/// ...
struct SliceInfoBase
{
    int instanceNum = 0;
    double z = 0;
    int fileNum = 0;
    auto operator <=>( const SliceInfoBase & ) const = default;
};

/// ...
struct SliceInfo : SliceInfoBase
{
    // these fields will be ignored in sorting
    Vector3d imagePos;
};

/// ...
MRVOXELS_API void sortScansByOrder( std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder );

/// ...
MRVOXELS_API void putScanFileNameInZ( const std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder );

/// Sort files in given vector by names (respect numbers in it)
/// usually needed for scans
MRVOXELS_API void sortScanFilesByName( std::vector<std::filesystem::path>& scans );

} // namespace MR
