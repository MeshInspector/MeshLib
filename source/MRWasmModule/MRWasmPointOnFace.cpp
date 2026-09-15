#include "MRMesh/MRPointOnFace.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace Wasm
{
EMSCRIPTEN_DECLARE_VAL_TYPE( PointOnFaceVal )
}

EMSCRIPTEN_BINDINGS( meshlib_point_on_face )
{
    emscripten::register_type<Wasm::PointOnFaceVal>( "PointOnFace",
        "{\n"
        "  face: number;\n"
        "  point: Vector3f;\n"
        "}" );
}
