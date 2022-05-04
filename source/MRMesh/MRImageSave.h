#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include <tl/expected.hpp>
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
MRMESH_API tl::expected<void, std::string> toBmp( const Image& image, const std::filesystem::path& path );

#ifndef __EMSCRIPTEN__
/// saves in .png format
MRMESH_API tl::expected<void, std::string> toPng( const Image& image, const std::filesystem::path& path );

/// saves in .png format
MRMESH_API tl::expected<void, std::string> toPng( const Image& image, std::ostream& os );

/// saves in .jpg format
MRMESH_API tl::expected<void, std::string> toJpeg( const Image& image, const std::filesystem::path& path );
#endif

/// detects the format from file extension and save image to it  
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Image& image, const std::filesystem::path& path );

/// \}

}

}
