#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <ostream>
#include <string>
#include "MRIOFilters.h"
#include "MRProgressCallback.h"

namespace MR
{

namespace LinesSave
{

/// \defgroup LinesSaveGroup Lines Save
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// saves in .mrlines file
MRMESH_API tl::expected<void, std::string> toMrLines( const Polyline3& polyline, const std::filesystem::path& file, ProgressCallback callback = emptyProgressCallback );
MRMESH_API tl::expected<void, std::string> toMrLines( const Polyline3& polyline, std::ostream& out, ProgressCallback callback = emptyProgressCallback );

/// detects the format from file extension and saves polyline in it
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file,
                                                                 ProgressCallback callback = emptyProgressCallback );
/// extension in `*.ext` format
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension,
                                                                 ProgressCallback callback = emptyProgressCallback );

} // namespace LinesSave

} // namespace MR
