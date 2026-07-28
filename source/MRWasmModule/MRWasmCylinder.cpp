#include "MRWasmBindings.h"

#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_cylinder )
{
    emscripten::function( "makeCylinderAdvanced",
        +[]( float radius0, float radius1, float start_angle, float arc_size, float length, int resolution )
    {
        return std::make_shared<Mesh>(
            makeCylinderAdvanced( radius0, radius1, start_angle, arc_size, length, resolution ) );
    } );
}
