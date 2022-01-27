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

MRMESH_API extern const IOFilters Filters;

// saves in .bmp format
MRMESH_API tl::expected<void, std::string> toBmp( const Image& image, const std::filesystem::path& path );

// detects the format from file extension and save image to it  
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Image& image, const std::filesystem::path& path );

}
}
