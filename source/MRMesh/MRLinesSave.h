#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <ostream>
#include <string>
#include "MRIOFilters.h"

namespace MR
{

namespace LinesSave
{

MRMESH_API extern const IOFilters Filters;

// saves in .mrlines file
MRMESH_API tl::expected<void, std::string> toMrLines( const Polyline3& polyline, const std::filesystem::path& file );
MRMESH_API tl::expected<void, std::string> toMrLines( const Polyline3& polyline, std::ostream& out );

// detects the format from file extension and saves polyline in it
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file );
// extension in `*.ext` format
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension );

} //namespace LinesSave

} //namespace MR
