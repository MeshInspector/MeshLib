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
    std::shared_ptr<AffineXf3f> xf;
};
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
/// important on Windows: in stream must be open in binary mode
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );

MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneGltfFile( const std::filesystem::path& file, bool combineAllObjects,
    ProgressCallback callback = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
