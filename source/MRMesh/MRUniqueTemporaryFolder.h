#pragma once

#include "MRMeshFwd.h"

#include <filesystem>

namespace MR
{

/// this callback will be called before compression on serialization and after decompression on deserialization
using FolderCallback = std::function<void( const std::filesystem::path& tempFolderName )>;

/// helper class to create a temporary folder; the folder will be removed on the object's destruction
class UniqueTemporaryFolder
{
public:
    /// creates new folder in temp directory
    MRMESH_API UniqueTemporaryFolder( FolderCallback onPreTempFolderDelete );
    /// removes folder with all its content
    MRMESH_API ~UniqueTemporaryFolder();

    explicit operator bool() const
    {
        return !folder_.empty();
    }
    operator const std::filesystem::path& ( ) const
    {
        return folder_;
    }
    std::filesystem::path operator /( const std::filesystem::path& child ) const
    {
        return folder_ / child;
    }

private:
    std::filesystem::path folder_;
    FolderCallback onPreTempFolderDelete_;
};

} // namespace MR
