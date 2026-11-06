#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_DRACO
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshLoadSettings.h>
#include <MRMesh/MRPointsLoadSettings.h>
#include <MRMesh/MRSaveSettings.h>

#include <filesystem>

namespace MR
{

/// options for saving in Google Draco format
struct DracoSaveOptions : SaveSettings
{
    /// 0 - fastest encoding and decoding but the worst compression, 10 - the best compression
    int compressionLevel = 7;
    /// the number of bits to quantize vertex coordinates to (in the range [1, 30]);
    /// 0 - no quantization, the coordinates are saved losslessly
    int positionQuantizationBits = 0;
};

namespace MeshLoad
{

/// loads from Google Draco .drc file;
/// the vertices with exactly the same coordinates are merged;
/// if the file contains a point cloud, then the mesh without triangles is returned
MRIOEXTRAS_API Expected<Mesh> fromDrc( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<Mesh> fromDrc( std::istream& in, const MeshLoadSettings& settings = {} );

} // namespace MeshLoad

namespace MeshSave
{

/// saves in Google Draco .drc file
MRIOEXTRAS_API Expected<void> toDrc( const Mesh& mesh, const std::filesystem::path& file, const DracoSaveOptions& options );
MRIOEXTRAS_API Expected<void> toDrc( const Mesh& mesh, std::ostream& out, const DracoSaveOptions& options );
MRIOEXTRAS_API Expected<void> toDrc( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API Expected<void> toDrc( const Mesh& mesh, std::ostream& out, const SaveSettings& settings = {} );

} // namespace MeshSave

namespace PointsLoad
{

/// loads from Google Draco .drc file; if the file contains a mesh, then its points are returned without triangles and without merging
MRIOEXTRAS_API Expected<PointCloud> fromDrc( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<PointCloud> fromDrc( std::istream& in, const PointsLoadSettings& settings = {} );

} // namespace PointsLoad

namespace PointsSave
{

/// saves in Google Draco .drc file
MRIOEXTRAS_API Expected<void> toDrc( const PointCloud& points, const std::filesystem::path& file, const DracoSaveOptions& options );
MRIOEXTRAS_API Expected<void> toDrc( const PointCloud& points, std::ostream& out, const DracoSaveOptions& options );
MRIOEXTRAS_API Expected<void> toDrc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings = {} );
MRIOEXTRAS_API Expected<void> toDrc( const PointCloud& points, std::ostream& out, const SaveSettings& settings = {} );

} // namespace PointsSave

} // namespace MR
#endif
