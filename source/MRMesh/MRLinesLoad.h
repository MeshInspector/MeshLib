#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace LinesLoad
{

/// \defgroup LinesLoad Lines Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// loads from .mrlines file
MRMESH_API tl::expected<Polyline3, std::string> fromMrLines( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API tl::expected<Polyline3, std::string> fromMrLines( std::istream& in, ProgressCallback callback = {} );

/// loads from .pts file
MRMESH_API tl::expected<Polyline3, std::string> fromPts( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API tl::expected<Polyline3, std::string> fromPts( std::istream& in, ProgressCallback callback = {} );

/// detects the format from file extension and loads polyline from it
MRMESH_API tl::expected<Polyline3, std::string> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API tl::expected<Polyline3, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback = {} );

/// \}

} // namespace LinesLoad

} // namespace MR
