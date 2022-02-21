#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <istream>
#include <string>
#include "MRPointCloud.h"
#include "MRIOFilters.h"

namespace MR
{

namespace PointsLoad
{

MRMESH_API extern const IOFilters Filters;

// loads from .ctm file
MRMESH_API tl::expected<PointCloud, std::string> fromCtm( const std::filesystem::path& file, std::vector<Color>* colors = nullptr );
MRMESH_API tl::expected<PointCloud, std::string> fromCtm( std::istream& in, std::vector<Color>* colors = nullptr );

// loads from .ply file
MRMESH_API tl::expected<PointCloud, std::string> fromPly( const std::filesystem::path& file, std::vector<Color>* colors = nullptr );
MRMESH_API tl::expected<PointCloud, std::string> fromPly( std::istream& in, std::vector<Color>* colors = nullptr );

// loads from .pts file
MRMESH_API tl::expected<PointCloud, std::string> fromPts( const std::filesystem::path& file );
MRMESH_API tl::expected<PointCloud, std::string> fromPts( std::istream& in );

// loads from .obj file
MRMESH_API tl::expected<PointCloud, std::string> fromObj( const std::filesystem::path& file );
MRMESH_API tl::expected<PointCloud, std::string> fromObj( std::istream& in );

// detects the format from file extension and loads points from it
MRMESH_API tl::expected<PointCloud, std::string> fromAnySupportedFormat( const std::filesystem::path& file, std::vector<Color>* colors = nullptr );

} //namespace PointsLoad

} //namespace MR
