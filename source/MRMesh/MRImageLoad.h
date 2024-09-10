#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

struct Image;

namespace ImageLoad
{

/// \defgroup ImageLoadGroup Image Load
/// \ingroup IOGroup
/// \{

/// detects the format from file extension and loads image from it
MRMESH_API Expected<Image> fromAnySupportedFormat( const std::filesystem::path& path );

/// \}

}

}
