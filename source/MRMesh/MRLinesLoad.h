#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRLinesLoadSettings.h"
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

/// loads polyline from file in internal MeshLib format
MRMESH_API Expected<Polyline3> fromMrLines( const std::filesystem::path& file, const LinesLoadSettings& settings = {} );

/// loads polyline from stream in internal MeshLib format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Polyline3> fromMrLines( std::istream& in, const LinesLoadSettings& settings = {} );

/// loads polyline from file in .PTS format
MRMESH_API Expected<Polyline3> fromPts( const std::filesystem::path& file, const LinesLoadSettings& settings = {} );

/// loads polyline from stream in .PTS format
MRMESH_API Expected<Polyline3> fromPts( std::istream& in, const LinesLoadSettings& settings = {} );

/// loads polyline from file in .PLY format
MRMESH_API Expected<Polyline3> fromPly( const std::filesystem::path& file, const LinesLoadSettings& settings = {} );

/// loads polyline from stream in .PLY format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Polyline3> fromPly( std::istream& in, const LinesLoadSettings& settings = {} );

/// loads polyline from file in the format detected from file extension
MRMESH_API Expected<Polyline3> fromAnySupportedFormat( const std::filesystem::path& file, const LinesLoadSettings& settings = {} );

/// loads polyline from stream in the format detected from given extension-string (`*.ext`);
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Polyline3> fromAnySupportedFormat( std::istream& in, const std::string& extension, const LinesLoadSettings& settings = {} );

/// \}

} // namespace LinesLoad

} // namespace MR
