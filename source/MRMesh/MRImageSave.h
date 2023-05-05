#pragma once

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

MRMESH_API extern const IOFilters Filters;

/// saves in .bmp format
MRMESH_API VoidOrErrStr toBmp( const Image& image, const std::filesystem::path& path );

#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_PNG
/// saves in .png format
MRMESH_API VoidOrErrStr toPng( const Image& image, const std::filesystem::path& path );

/// saves in .png format
MRMESH_API VoidOrErrStr toPng( const Image& image, std::ostream& os );
#endif

#ifndef MRMESH_NO_JPEG
/// saves in .jpg format
MRMESH_API VoidOrErrStr toJpeg( const Image& image, const std::filesystem::path& path );
#endif

#ifndef MRMESH_NO_TIFF
MRMESH_API VoidOrErrStr toTiff( const Image& image, const std::filesystem::path& path );
#endif

#endif

/// detects the format from file extension and save image to it  
MRMESH_API VoidOrErrStr toAnySupportedFormat( const Image& image, const std::filesystem::path& path );

/// \}

}

}
