#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <filesystem>
#include <iosfwd>
#include <string>

namespace MR
{

namespace GcodeSave
{

/// \defgroup GcodeSaveGroup Gcode Save
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// saves G-code commands in *.gcode file (or any text file), one command per line
MRMESH_API Expected<void> toGcode( const GcodeSource& gcode, const std::filesystem::path& file, ProgressCallback callback = {} );

MRMESH_API Expected<void> toGcode( const GcodeSource& gcode, std::ostream& out, ProgressCallback callback = {} );


/// detects the format from file extension and saves G-code in it
MRMESH_API Expected<void> toAnySupportedFormat( const GcodeSource& gcode, const std::filesystem::path& file, ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API Expected<void> toAnySupportedFormat( const GcodeSource& gcode, std::ostream& out, const std::string& extension, ProgressCallback callback = {} );

/// \}

}

}
