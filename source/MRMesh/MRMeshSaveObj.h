#pragma once

#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRExpected.h"
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
MRMESH_API Expected<void> sceneToObj( const std::vector<NamedXfMesh> & objects, const std::filesystem::path & file,
                                    VertColors* colors = nullptr );
MRMESH_API Expected<void> sceneToObj( const std::vector<NamedXfMesh> & objects, std::ostream & out,
                                    VertColors* colors = nullptr );

/// \}

} // namespace MeshSave

} // namespace MR
