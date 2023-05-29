#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include <tl/expected.hpp>
#include <filesystem>

namespace MR
{

namespace GcodeLoad
{

/// \defgroup GcodeLoadGroup Mesh Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// loads from *.gcode file (or any text file)
MRMESH_API tl::expected<GcodeSource, std::string> fromGcode( const std::filesystem::path& file, ProgressCallback callback = {} );

/// detects the format from file extension and loads mesh from it
MRMESH_API tl::expected<GcodeSource, std::string> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback = {} );

/// \}

}

}
