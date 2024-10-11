#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

struct Image;

namespace ImageSave
{

/// \defgroup ImageSaveGroup Image Save
/// \ingroup IOGroup
/// \{

/// saves in .bmp format
MRMESH_API Expected<void> toBmp( const Image& image, const std::filesystem::path& path );

#ifndef __EMSCRIPTEN__

#ifndef MRMESH_NO_TIFF
MRMESH_API Expected<void> toTiff( const Image& image, const std::filesystem::path& path );
#endif

#endif

/// detects the format from file extension and save image to it  
MRMESH_API Expected<void> toAnySupportedFormat( const Image& image, const std::filesystem::path& path );

/// \}

}

}
