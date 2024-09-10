#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_LAS
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRPointsLoadSettings.h>

#include <filesystem>

namespace MR
{

namespace PointsLoad
{

/// loads from .las file
MRIOEXTRAS_API Expected<PointCloud> fromLas( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<PointCloud> fromLas( std::istream& in, const PointsLoadSettings& settings = {} );

} // namespace PointsLoad

} // namespace MR
#endif
