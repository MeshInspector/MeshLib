#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRRaster.h"

#include <filesystem>

namespace MR
{

namespace RasterLoad
{

/// \defgroup RasterLoadGroup Raster Load
/// \ingroup IOGroup
/// \{

/// detects the format from file extension and loads a raster from it
MRMESH_API Expected<Raster> fromAnySupportedFormat( const std::filesystem::path& path, const RasterLoadSettings& settings = {} );

/// detects the format from file extension and loads everything about the raster except its samples
MRMESH_API Expected<RasterInfo> infoFromAnySupportedFormat( const std::filesystem::path& path );

/// \}

} // namespace RasterLoad

} // namespace MR
