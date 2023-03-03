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
    std::string materialLibrary;
    struct MaterialVertMap
    {
        std::string materialName;
        VertBitSet vertices;
    };
    std::vector<MaterialVertMap> materialVertMaps;
};
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
/// important on Windows: in stream must be open in binary mode
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );
MRMESH_API tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects,
                                                                               ProgressCallback callback = {} );

/// load material definitions from Material Library File (.mtl)
struct MtlMaterial
{
    Vector3f diffuseColor;
    std::string diffuseTextureFile;
};
using MtlLibrary = std::map<std::string, MtlMaterial>;
MRMESH_API tl::expected<MtlLibrary, std::string> loadMtlLibrary( const char* data, size_t size, ProgressCallback callback = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
