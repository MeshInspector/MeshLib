#pragma once

#pragma once

#include "MRMesh.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace MeshLoad
{

// loads scene from obj file
struct NamedMesh
{
    std::string name;
    Mesh mesh;
};
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects );
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects );

} //namespace MeshLoad

} //namespace MR
