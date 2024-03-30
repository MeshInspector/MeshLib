#pragma once

#include "MRMesh.h"
#include "MRMeshTexture.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
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
    VertUVCoords uvCoords;
    VertColors colors;
    std::filesystem::path pathToTexture;
    std::optional<Color> diffuseColor;
};
MRMESH_API Expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                           const MeshLoadSettings& settings = {} );
/// important on Windows: in stream must be open in binary mode
/// \param dir working directory where materials and textures are located
MRMESH_API Expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                           const MeshLoadSettings& settings = {} );
/// \param dir working directory where materials and textures are located
MRMESH_API Expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                           const MeshLoadSettings& settings = {} );
} // namespace MeshLoad

} // namespace MR
