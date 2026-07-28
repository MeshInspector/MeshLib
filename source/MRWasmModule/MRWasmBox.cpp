#include "MRWasmBindings.h"

#include "MRMesh/MRBox.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_box )
{
    emscripten::class_<Box3f>( "Box3f" )
        .constructor<>()
        .constructor<const Vector3f&, const Vector3f&>()
        .property( "min", &Box3f::min )
        .property( "max", &Box3f::max )
        .function( "valid", &Box3f::valid )
        .function( "size", &Box3f::size )
        .function( "diagonal", &Box3f::diagonal )
        .function( "volume", &Box3f::volume )
        .function( "center", &Box3f::center );

    emscripten::class_<Box3i>( "Box3i" )
        .constructor<>()
        .constructor<const Vector3i&, const Vector3i&>()
        .property( "min", &Box3i::min )
        .property( "max", &Box3i::max )
        .function( "valid", &Box3i::valid )
        .function( "size", &Box3i::size )
        .function( "volume", &Box3i::volume );
}
