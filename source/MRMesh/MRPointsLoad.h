#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRPointsLoadSettings.h"

#include <filesystem>

namespace MR::PointsLoad
{

/// \defgroup PointsLoadGroup Points Load
/// \addtogroup IOGroup
/// \{

/// loads from .csv, .asc, .xyz, .txt file
MRMESH_API Expected<PointCloud> fromText( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromText( std::istream& in, const PointsLoadSettings& settings = {} );

/// loads from Laser scan plain data format (.pts) file
MRMESH_API Expected<PointCloud> fromPts( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromPts( std::istream& in, const PointsLoadSettings& settings = {} );

/// loads from .ply file
MRMESH_API Expected<PointCloud> fromPly( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromPly( std::istream& in, const PointsLoadSettings& settings = {} );

/// loads from .obj file
MRMESH_API Expected<PointCloud> fromObj( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromObj( std::istream& in, const PointsLoadSettings& settings = {} );

MRMESH_API Expected<PointCloud> fromDxf( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromDxf( std::istream& in, const PointsLoadSettings& settings = {} );

/// detects the format from file extension and loads points from it
MRMESH_API Expected<PointCloud> fromAnySupportedFormat( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<PointCloud> fromAnySupportedFormat( std::istream& in, const std::string& extension, const PointsLoadSettings& settings = {} );

/// emits telemetry signal with the integer logarithm of the number of points
MR_BIND_IGNORE MRMESH_API void telemetryLogSize( const PointCloud& cloud );

/// loads and merges a point cloud from a folder produced by a multi-scan laser capture:
/// the folder is expected to contain pairs of files named _intempNNN.pose and _laserNNN.ply with matching indices NNN
/// (the number of pairs is arbitrary); each .pose file stores a 4x4 row-major rigid transformation,
/// which is applied to the points (and normals) loaded from the .ply file with the same index;
/// \return the union of all transformed points from all found pairs, expressed in the common coordinate frame
MRMESH_API Expected<PointCloud> fromMultiScanFolder( const std::filesystem::path& folder, const ProgressCallback& callback = {} );

/// \}

} // namespace MR::PointsLoad
