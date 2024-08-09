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

namespace GcodeLoad
{

/// \defgroup GcodeLoadGroup Mesh Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// loads from *.gcode file (or any text file)
MRMESH_API Expected<GcodeSource> fromGcode( const std::filesystem::path& file, ProgressCallback callback = {} );

MRMESH_API Expected<GcodeSource> fromGcode( std::istream& in, ProgressCallback callback = {} );


/// detects the format from file extension and loads mesh from it
MRMESH_API Expected<GcodeSource> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API Expected<GcodeSource> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback = {} );

/// \}

}

}
