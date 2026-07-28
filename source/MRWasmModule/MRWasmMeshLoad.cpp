#include "MRWasmBindings.h"

#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <memory>
#include <string>

using namespace MR;

namespace
{
struct MeshLoadModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_mesh_load )
{
    emscripten::class_<MeshLoadModule>( "MeshLoad" )
        .class_function( "fromAnySupportedFormat", +[]( const std::string& path ) -> std::shared_ptr<Mesh>
        {
            return std::make_shared<Mesh>( Wasm::unwrap( MeshLoad::fromAnySupportedFormat( std::filesystem::path( path ) ) ) );
        } );
}
