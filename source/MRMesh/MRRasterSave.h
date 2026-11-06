#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRRaster.h"

#include <filesystem>

namespace MR
{

namespace RasterSave
{

/// \defgroup RasterSaveGroup Raster Save
/// \ingroup IOGroup
/// \{

/// detects the format from file extension and saves the raster in it
MRMESH_API Expected<void> toAnySupportedFormat( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings = {} );

/// \}

} // namespace RasterSave

} // namespace MR
