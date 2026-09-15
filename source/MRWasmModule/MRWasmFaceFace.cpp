#include "MRMesh/MRFaceFace.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace Wasm
{
EMSCRIPTEN_DECLARE_VAL_TYPE( FaceFaceVal )
}

EMSCRIPTEN_BINDINGS( meshlib_face_face )
{
    emscripten::register_type<Wasm::FaceFaceVal>( "FaceFace", "{ aFace: number; bFace: number }" );
}
