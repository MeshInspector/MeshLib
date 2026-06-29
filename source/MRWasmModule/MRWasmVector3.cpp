#include "MRWasmBindings.h"

#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_vector3 )
{
    emscripten::value_object<Vector3f>( "Vector3f" )
        .field( "x", &Vector3f::x )
        .field( "y", &Vector3f::y )
        .field( "z", &Vector3f::z );

    emscripten::value_object<Vector3i>( "Vector3i" )
        .field( "x", &Vector3i::x )
        .field( "y", &Vector3i::y )
        .field( "z", &Vector3i::z );
}
