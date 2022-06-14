#pragma once

#include "MRMesh.h"
#include "MRAffineXf3.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <ostream>
#include <string>

namespace MR
{

namespace MeshSave
{

/// \defgroup MeshSaveObjGroup Mesh Save Obj
/// \ingroup IOGroup
/// \{

/// saves a number of named meshes in .obj file
struct NamedXfMesh
{
    std::string name;
    AffineXf3f toWorld;
    std::shared_ptr<const Mesh> mesh;
};
MRMESH_API tl::expected<void, std::string> sceneToObj( const std::vector<NamedXfMesh> & objects, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> sceneToObj( const std::vector<NamedXfMesh> & objects, std::ostream & out );

/// \}

} // namespace MeshSave

} // namespace MR
