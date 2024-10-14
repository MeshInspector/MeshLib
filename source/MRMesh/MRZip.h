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

/**
 * \brief compresses given folder in given zip-file
 * \param excludeFiles files that should not be included to result zip 
 * \param password if password is given then the archive will be encrypted
 * \param cb an option to get progress notifications and cancel the operation
 */
MRMESH_API Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder, 
    const std::vector<std::filesystem::path>& excludeFiles = {}, const char * password = nullptr, ProgressCallback cb = {} );

/// \}

} // namespace MR
