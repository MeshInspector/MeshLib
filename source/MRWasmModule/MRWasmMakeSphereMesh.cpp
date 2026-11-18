#include "MRWasmBindings.h"

#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_make_sphere_mesh )
{
    emscripten::value_object<SphereParams>( "SphereParams" )
        .field( "radius", &SphereParams::radius )
        .field( "numMeshVertices", &SphereParams::numMeshVertices );

    emscripten::function( "makeSphere", +[]( const SphereParams& params )
    {
        return std::make_shared<Mesh>( makeSphere( params ) );
    } );

    emscripten::function( "makeUVSphere",
        +[]( float radius, int horisontalResolution, int verticalResolution )
    {
        return std::make_shared<Mesh>( makeUVSphere( radius, horisontalResolution, verticalResolution ) );
    } );
}
