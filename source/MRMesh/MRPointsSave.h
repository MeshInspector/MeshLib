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

MRMESH_API extern const IOFilters Filters;

/// save valid points with normals in textual .asc file
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

/// saves in .ply file
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

struct CtmSavePointsOptions : SaveSettings
{
    /// 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1;
    /// comment saved in the file
    const char* comment = "MeshInspector Points";
};

#ifndef MRMESH_NO_OPENCTM
/// saves in .ctm file
MRMESH_API VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const CtmSavePointsOptions& options = {} );
MRMESH_API VoidOrErrStr toCtm( const PointCloud& points, std::ostream& out, const CtmSavePointsOptions& options = {} );
#endif

/// detects the format from file extension and save points to it
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const SaveSettings& settings = {} );

/// \}

} // namespace PointsSave

} // namespace MR
