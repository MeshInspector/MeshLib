#include "MRWasmBindings.h"

#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_matrix3 )
{
    emscripten::class_<Matrix3f>( "Matrix3f" )
        .constructor<>()
        .constructor<const Vector3f&, const Vector3f&, const Vector3f&>()
        .property( "x", &Matrix3f::x )
        .property( "y", &Matrix3f::y )
        .property( "z", &Matrix3f::z )
        .class_function( "identity", &Matrix3f::identity )
        .class_function( "zero", &Matrix3f::zero )
        .class_function( "rotation",
            +[]( const Vector3f& axis, float angle ) { return Matrix3f::rotation( axis, angle ); } );
}
