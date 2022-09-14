#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include <tl/expected.hpp>
#include <filesystem>

namespace MR
{

struct Image;

namespace ImageLoad
{

/// \defgroup ImageLoadGroup Image Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

#ifndef MRMESH_NO_PNG
/// loads from .png format
MRMESH_API tl::expected<Image, std::string> fromPng( const std::filesystem::path& path );
#endif

/// detects the format from file extension and loads image from it
MRMESH_API tl::expected<Image, std::string> fromAnySupportedFormat( const std::filesystem::path& path );

/// \}

}

}
