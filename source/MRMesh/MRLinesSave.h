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

/// saves in .mrlines file;
/// SaveSettings::onlyValidPoints = true is ignored
MRMESH_API Expected<void> toMrLines( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toMrLines( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .pts file;
/// SaveSettings::onlyValidPoints = false is ignored
MRMESH_API Expected<void> toPts( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toPts( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .dxf file;
/// SaveSettings::onlyValidPoints = false is ignored
MRMESH_API Expected<void> toDxf( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toDxf( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .ply file
MRMESH_API Expected<void> toPly( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toPly( const Polyline3& polyline, std::ostream & out, const SaveSettings & settings = {} );

/// detects the format from file extension and saves polyline in it
MRMESH_API Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::string& extension, std::ostream& out, const SaveSettings & settings = {} );

} // namespace LinesSave

} // namespace MR
