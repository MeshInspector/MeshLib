#pragma once

#include "MRMesh.h"
#include "MRProgressCallback.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace MeshLoad
{

/// \defgroup MeshLoadObjGroup Mesh Load Obj
/// \ingroup IOGroup
/// \{

/// loads scene from obj file
struct NamedMesh
{
    std::string name;
    Mesh mesh;
};
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
