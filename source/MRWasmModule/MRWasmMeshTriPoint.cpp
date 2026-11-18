#include "MRWasmMeshTriPoint.h"

#include "MRMesh/MRMeshTriPoint.h"

#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS( meshlib_mesh_tri_point )
{
    emscripten::register_type<Wasm::MeshTriPointVal>( "MeshTriPoint",
        "{\n"
        "  e: number;\n"
        "  bary: { a: number; b: number };\n"
        "}" );
}
