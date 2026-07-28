#include "MRWasmBindings.h"

#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

namespace
{
struct Vector3Module {};
}

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

    emscripten::class_<Vector3Module>( "Vector3" )
        .class_function( "plusZ", +[]() { return Vector3f::plusZ(); } )
        .class_function( "diagonal", +[]( float a ) { return Vector3f::diagonal( a ); } )
        .class_function( "add", +[]( const Vector3f& a, const Vector3f& b ) { return a + b; } )
        .class_function( "mulScalar", +[]( const Vector3f& v, float s ) { return v * s; } );
}
