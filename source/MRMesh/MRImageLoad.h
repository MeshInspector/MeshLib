#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRExpected.h"
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
MRMESH_API Expected<Image> fromPng( const std::filesystem::path& path );
MRMESH_API Expected<Image> fromPng( std::istream& in );
#endif

#ifndef MRMESH_NO_JPEG
/// loads from .jpg format
MRMESH_API Expected<Image> fromJpeg( const std::filesystem::path& path );
MRMESH_API Expected<Image> fromJpeg( std::istream& in );
MRMESH_API Expected<Image> fromJpeg( const char* data, size_t size );
#endif

/// detects the format from file extension and loads image from it
MRMESH_API Expected<Image> fromAnySupportedFormat( const std::filesystem::path& path );

/// \}

}

}
