#include "MRWasmBindings.h"

#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_collide )
{
    emscripten::function( "isInside", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        return isInside( *a, *b );
    } );
}
