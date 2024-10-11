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

namespace MeshSave
{

struct CtmSaveOptions : SaveSettings
{
    enum class MeshCompression
    {
        None,     ///< no compression at all, fast but not effective
        Lossless, ///< compression without any loss in vertex coordinates
        Lossy     ///< compression with loss in vertex coordinates
    };
    MeshCompression meshCompression = MeshCompression::Lossless;
    /// fixed point precision for vertex coordinates in case of MeshCompression::Lossy.
    /// For instance, if this value is 0.001, all vertex coordinates will be rounded to three decimals
    float vertexPrecision = 1.0f / 1024.0f; //~= 0.00098
    /// LZMA compression: 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1;
    /// comment saved in the file
    const char * comment = "MeshInspector.com";
};

/// saves in .ctm file
MRIOEXTRAS_API Expected<void> toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions & options );
MRIOEXTRAS_API Expected<void> toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions & options );
MRIOEXTRAS_API Expected<void> toCtm( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
MRIOEXTRAS_API Expected<void> toCtm( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

} // namespace MeshSave

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
MRIOEXTRAS_API Expected<void> toCtm( const PointCloud& points, const std::filesystem::path& file, const CtmSavePointsOptions& options );
MRIOEXTRAS_API Expected<void> toCtm( const PointCloud& points, std::ostream& out, const CtmSavePointsOptions& options );
MRIOEXTRAS_API Expected<void> toCtm( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API Expected<void> toCtm( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

} // namespace PointsSave

} // namespace MR
#endif
