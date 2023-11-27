#pragma once

#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRSaveSettings.h"
#include <filesystem>
#include <ostream>

namespace MR
{

namespace LinesSave
{

/// \defgroup LinesSaveGroup Lines Save
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// saves in .mrlines file;
/// SaveSettings::saveValidOnly = true is ignored
MRMESH_API VoidOrErrStr toMrLines( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toMrLines( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .pts file;
/// SaveSettings::saveValidOnly = false is ignored
MRMESH_API VoidOrErrStr toPts( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toPts( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .dxf file;
/// SaveSettings::saveValidOnly = false is ignored
MRMESH_API VoidOrErrStr toDxf( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toDxf( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// detects the format from file extension and saves polyline in it
MRMESH_API VoidOrErrStr toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension, const SaveSettings & settings = {} );

} // namespace LinesSave

} // namespace MR
