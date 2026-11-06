#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_LAS
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRPointsLoadSettings.h>
#include <MRMesh/MRSaveSettings.h>

#include <filesystem>

namespace MR
{

namespace PointsLoad
{

/// loads from .las file
MRIOEXTRAS_API Expected<PointCloud> fromLas( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<PointCloud> fromLas( std::istream& in, const PointsLoadSettings& settings = {} );

} // namespace PointsLoad

namespace PointsSave
{

/// saves in uncompressed .las file (LAS 1.2, point format 0 or 2 if colors are given);
/// normals are saved as three double extra bytes NormalX, NormalY, NormalZ
MRIOEXTRAS_API Expected<void> toLas( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API Expected<void> toLas( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// saves in compressed .laz file, the same contents as in toLas
MRIOEXTRAS_API Expected<void> toLaz( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API Expected<void> toLaz( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

} // namespace PointsSave

} // namespace MR
#endif
