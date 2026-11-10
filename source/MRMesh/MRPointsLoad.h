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

/// loads from Point Cloud Data (.pcd) file with ascii, binary or binary_compressed data;
/// reads coordinates, normals (normal_x, normal_y, normal_z fields) and colors (rgb or rgba field), points with not-finite coordinates are invalid
MRMESH_API Expected<PointCloud> fromPcd( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromPcd( std::istream& in, const PointsLoadSettings& settings = {} );

/// loads from Leica .ptx file, where every scan is moved in the common coordinates by its matrix;
/// the directions without reflection (stored as zero points) are invalid in the result
MRMESH_API Expected<PointCloud> fromPtx( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
MRMESH_API Expected<PointCloud> fromPtx( std::istream& in, const PointsLoadSettings& settings = {} );

/// detects the format from file extension and loads points from it
MR_BIND_OVERLOAD_SUFFIX() MRMESH_API Expected<PointCloud> fromAnySupportedFormat( const std::filesystem::path& file, const PointsLoadSettings& settings = {} );
/// extension in `*.ext` format
MR_BIND_OVERLOAD_SUFFIX(stream) MRMESH_API Expected<PointCloud> fromAnySupportedFormat( std::istream& in, const std::string& extension, const PointsLoadSettings& settings = {} );

/// emits telemetry signal with the integer logarithm of the number of points
MR_BIND_IGNORE MRMESH_API void telemetryLogSize( const PointCloud& cloud );

/// \}

} // namespace MR::PointsLoad
