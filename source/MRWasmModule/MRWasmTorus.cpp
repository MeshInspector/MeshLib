#include "MRWasmBindings.h"

#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_torus )
{
    emscripten::function( "makeTorus",
        +[]( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution )
    {
        return std::make_shared<Mesh>(
            makeTorus( primaryRadius, secondaryRadius, primaryResolution, secondaryResolution ) );
    } );

    emscripten::function( "makeTorusWithSelfIntersections",
        +[]( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution )
    {
        return std::make_shared<Mesh>(
            makeTorusWithSelfIntersections( primaryRadius, secondaryRadius, primaryResolution, secondaryResolution ) );
    } );
}
