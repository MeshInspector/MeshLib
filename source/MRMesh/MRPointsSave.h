#pragma once

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

/// saves in .ply file
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors = nullptr,
                                                  ProgressCallback callback = {} );
MRMESH_API VoidOrErrStr toPly( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors = nullptr,
                                                  ProgressCallback callback = {} );

struct CtmSavePointsOptions
{
    /// 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1;
    /// comment saved in the file
    const char* comment = "MeshInspector Points";
};

#ifndef MRMESH_NO_OPENCTM
/// saves in .ctm file
MRMESH_API VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors = nullptr,
                                                  const CtmSavePointsOptions& options = {}, ProgressCallback callback = {} );
MRMESH_API VoidOrErrStr toCtm( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors = nullptr,
                                                  const CtmSavePointsOptions& options = {}, ProgressCallback callback = {} );
#endif

/// detects the format from file extension and save points to it
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors = nullptr,
                                                                 ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const Vector<Color, VertId>* colors = nullptr,
                                                                 ProgressCallback callback = {} );

/// \}

}

}
