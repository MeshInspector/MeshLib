#include "MRWasmBindings.h"

#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_cube )
{
    emscripten::function( "makeCube", +[]()
    {
        return std::make_shared<Mesh>( makeCube() );
    } );
    emscripten::function( "makeCube", +[]( const Vector3f& size )
    {
        return std::make_shared<Mesh>( makeCube( size ) );
    } );
    emscripten::function( "makeCube", +[]( const Vector3f& size, const Vector3f& base )
    {
        return std::make_shared<Mesh>( makeCube( size, base ) );
    } );
}
