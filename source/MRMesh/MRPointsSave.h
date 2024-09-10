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

/// save valid points with normals in textual .asc file
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// saves in .ply file
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// detects the format from file extension and save points to it
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::string& extension, std::ostream& out, const SaveSettings& settings = {} );

/// \}

} // namespace PointsSave

} // namespace MR
