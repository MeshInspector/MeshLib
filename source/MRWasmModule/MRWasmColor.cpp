#include "MRWasmBindings.h"

#include "MRMesh/MRColor.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_color )
{
    emscripten::value_object<Color>( "Color" )
        .field( "r", &Color::r )
        .field( "g", &Color::g )
        .field( "b", &Color::b )
        .field( "a", &Color::a );
}
