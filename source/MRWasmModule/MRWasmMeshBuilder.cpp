#include "MRWasmBindings.h"

#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

namespace
{
struct MeshBuilderModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_mesh_builder )
{
    emscripten::class_<MeshBuilderModule>( "MeshBuilder" )
        .class_function( "uniteCloseVertices", +[]( std::shared_ptr<Mesh> m, float closeDist, bool uniteOnlyBd )
        {
            return MeshBuilder::uniteCloseVertices( *m, closeDist, uniteOnlyBd );
        } );
}
