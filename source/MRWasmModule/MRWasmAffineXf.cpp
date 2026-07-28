#include "MRWasmBindings.h"

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_affine_xf )
{
    emscripten::class_<AffineXf3f>( "AffineXf3f" )
        .constructor<>()
        .constructor<const Matrix3f&, const Vector3f&>()
        .property( "A", &AffineXf3f::A )
        .property( "b", &AffineXf3f::b )
        .class_function( "translation", &AffineXf3f::translation )
        .class_function( "linear", &AffineXf3f::linear )
        .function( "apply", +[]( const AffineXf3f& xf, const Vector3f& v ) { return xf( v ); } )
        .function( "mul", +[]( const AffineXf3f& a, const AffineXf3f& b ) { return a * b; } );
}
