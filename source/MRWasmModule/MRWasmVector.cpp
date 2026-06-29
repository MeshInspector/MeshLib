#include "MRWasmBindings.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>

#include <cstdint>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_vector )
{
    emscripten::class_<VertCoords>( "VertCoords" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<VertCoords, float, 3> )
        .function( "toArray", &Wasm::packedToTypedArray<VertCoords, float, 3> );

    emscripten::class_<Triangulation>( "Triangulation" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<Triangulation, uint32_t, 3> )
        .function( "toArray", &Wasm::packedToTypedArray<Triangulation, uint32_t, 3> );
}
