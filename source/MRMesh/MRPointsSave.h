#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRPointCloud.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
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

/// determines how to save point cloud
struct Settings
{
    /// true - save valid points only; false - save all points preserving their indices
    bool saveValidOnly = true;
    /// point colors to save if supported
    const VertColors* colors = nullptr;
    /// to report progress to the user and cancel saving
    ProgressCallback callback;
};

/// save valid points with normals in textual .asc file
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, const std::filesystem::path& file, const Settings& settings = {} );
MRMESH_API VoidOrErrStr toAsc( const PointCloud& points, std::ostream& out, const Settings& settings = {} );

/// saves in .ply file
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const Settings& settings = {} );
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, std::ostream& out, const Settings& settings = {} );

struct CtmSavePointsOptions : Settings
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
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const Settings& settings = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const Settings& settings = {} );

/// \}

} // namespace PointsSave

} // namespace MR
