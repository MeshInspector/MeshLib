#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_CTM
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshLoadSettings.h>
#include <MRMesh/MRPointsLoadSettings.h>
#include <MRMesh/MRSaveSettings.h>

#include <filesystem>

namespace MR
{

namespace MeshLoad
{

/// loads from .ctm file
MRIOEXTRAS_API Expected<Mesh> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<Mesh> fromCtm( std::istream& in, const MeshLoadSettings& settings = {} );

} // namespace MeshLoad

namespace PointsLoad
{

/// loads from .ctm file
MRIOEXTRAS_API Expected<PointCloud> fromCtm( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<PointCloud> fromCtm( std::istream& in, const PointsLoadSettings& settings = {} );

} // namespace PointsLoad

namespace PointsSave
{

struct CtmSavePointsOptions : SaveSettings
{
    /// 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1;
    /// comment saved in the file
    const char* comment = "MeshInspector Points";
};

/// saves in .ctm file
MRIOEXTRAS_API VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const CtmSavePointsOptions& options );
MRIOEXTRAS_API VoidOrErrStr toCtm( const PointCloud& points, std::ostream& out, const CtmSavePointsOptions& options );
MRIOEXTRAS_API VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API VoidOrErrStr toCtm( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

} // namespace PointsSave

} // namespace MR
#endif
