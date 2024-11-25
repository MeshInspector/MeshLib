#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRExpected.h>
#include <filesystem>

namespace MR
{

/// load all supported files from given folder in new container object
MRVIEWER_API Expected<Object> makeObjectTreeFromFolder( const std::filesystem::path& folder,
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

/// load all supported files from given zip-archive in new container object
MRVIEWER_API Expected<Object> makeObjectTreeFromZip( const std::filesystem::path& zipPath,
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

} //namespace MR
