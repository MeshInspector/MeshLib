#include "MRWasmBindings.h"

#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <memory>
#include <string>

using namespace MR;

namespace
{
struct MeshSaveModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_mesh_save )
{
    emscripten::class_<MeshSaveModule>( "MeshSave" )
        .class_function( "toAnySupportedFormat", +[]( std::shared_ptr<Mesh> mesh, const std::string& path )
        {
            Wasm::unwrap( MeshSave::toAnySupportedFormat( *mesh, std::filesystem::path( path ) ) );
        } );
}
