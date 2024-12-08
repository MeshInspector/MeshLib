#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRLoadedObjects.h>
#include <filesystem>

namespace MR
{

/// load all supported files from given folder in new container object
MRVIEWER_API Expected<LoadedObject> makeObjectTreeFromFolder( const std::filesystem::path& folder, const ProgressCallback& callback = {} );

/// load all supported files from given zip-archive in new container object
MRVIEWER_API Expected<LoadedObject> makeObjectTreeFromZip( const std::filesystem::path& zipPath, const ProgressCallback& callback = {} );

} //namespace MR
