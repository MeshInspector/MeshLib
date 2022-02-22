#pragma once

#include "MRMeshFwd.h"
#include <filesystem>
#include <tl/expected.hpp>

namespace MR
{

// loads mesh from given file in new object
MRMESH_API tl::expected<ObjectMesh, std::string> makeObjectMeshFromFile( const std::filesystem::path & file );

// loads lines from given file in new object
MRMESH_API tl::expected<ObjectLines, std::string> makeObjectLinesFromFile( const std::filesystem::path& file );

// loads points from given file in new object
MRMESH_API tl::expected<ObjectPoints, std::string> makeObjectPointsFromFile( const std::filesystem::path& file );

// loads meshes from given folder in new container object
MRMESH_API tl::expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path & folder );

} //namespace MR
