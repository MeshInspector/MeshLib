#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRPointCloud.h"
#include "MRIOFilters.h"
#include "MRSaveSettings.h"
#include <filesystem>
#include <ostream>

namespace MR
{

namespace PointsSave
{

/// \defgroup PointsSaveGroup Points Save
/// \ingroup IOGroup
/// \{

/// save points without normals in textual .xyz file;
/// each output line contains [x, y, z], where x, y, z are point coordinates
MRMESH_API Expected<void> toXyz( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API Expected<void> toXyz( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// save points with normals in textual .xyzn file;
/// each output line contains [x, y, z, nx, ny, nz], where x, y, z are point coordinates and nx, ny, nz are the components of point normal
MRMESH_API Expected<void> toXyzn( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API Expected<void> toXyzn( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// save points with normals in .xyzn format, and save points without normals in .xyz format
MRMESH_API Expected<void> toAsc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API Expected<void> toAsc( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// saves in .ply file
MRMESH_API Expected<void> toPly( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API Expected<void> toPly( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// detects the format from file extension and save points to it
MRMESH_API Expected<void> toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<void> toAnySupportedFormat( const PointCloud& points, const std::string& extension, std::ostream& out, const SaveSettings& settings = {} );

/// \}

} // namespace PointsSave

} // namespace MR
