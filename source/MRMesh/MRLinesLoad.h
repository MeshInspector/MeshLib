#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
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

/// loads from .mrlines file
MRMESH_API Expected<Polyline3> fromMrLines( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API Expected<Polyline3> fromMrLines( std::istream& in, ProgressCallback callback = {} );

/// loads from .pts file
MRMESH_API Expected<Polyline3> fromPts( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API Expected<Polyline3> fromPts( std::istream& in, ProgressCallback callback = {} );

/// detects the format from file extension and loads polyline from it
MRMESH_API Expected<Polyline3> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API Expected<Polyline3> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback = {} );

/// \}

} // namespace LinesLoad

} // namespace MR
