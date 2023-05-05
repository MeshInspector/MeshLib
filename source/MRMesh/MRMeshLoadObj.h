#pragma once

#include "MRMesh.h"
#include "MRMeshTexture.h"
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
    Vector<UVCoord, VertId> uvCoords;
    std::filesystem::path pathToTexture;
    std::optional<Color> diffuseColor;
};
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
/// important on Windows: in stream must be open in binary mode
/// \param dir working directory where materials and textures are located
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                               ProgressCallback callback = {} );
/// \param dir working directory where materials and textures are located
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                               ProgressCallback callback = {} );
} // namespace MeshLoad

} // namespace MR
