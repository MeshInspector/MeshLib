#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <filesystem>
#include <vector>

namespace MR
{

/// \defgroup SerializerGroup Serializer
/// \ingroup IOGroup
/// \{

/**
 * \brief decompresses given zip-file into given folder
 * \param password if password is given then it will be used to decipher encrypted archive
 */
MRMESH_API Expected<void> decompressZip( const std::filesystem::path& zipFile, const std::filesystem::path& targetFolder,
    const char * password = nullptr );

/**
 * \brief decompresses given binary stream (containing the data of a zip file only) into given folder
 * \param password if password is given then it will be used to decipher encrypted archive
 */
MRMESH_API Expected<void> decompressZip( std::istream& zipStream, const std::filesystem::path& targetFolder, const char * password = nullptr );

struct CompressZipSettings
{
    /// files that should not be included to result zip 
    std::vector<std::filesystem::path> excludeFiles;

    /// [0-9]: greater level means longer processing but better compression 
    /// 0 is special value to use default level
    int compressionLevel = 0;

    /// if password is given then the archive will be encrypted
    std::string password;

    /// an option to get progress notifications and cancel the operation
    ProgressCallback cb;
};

/**
 * \brief compresses given folder in given zip-file
 */
MRMESH_API Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder,
    const CompressZipSettings& settings );

/**
 * \brief compresses given folder in given zip-file
 * \param excludeFiles files that should not be included to result zip 
 * \param password if password is given then the archive will be encrypted
 * \param cb an option to get progress notifications and cancel the operation
 */
[[deprecated( "Use compressZip( zipFile, sourceFolder, settings )" )]]
MRMESH_API Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder, 
    const std::vector<std::filesystem::path>& excludeFiles = {}, const char * password = nullptr, ProgressCallback cb = {} );

/// \}

} // namespace MR
