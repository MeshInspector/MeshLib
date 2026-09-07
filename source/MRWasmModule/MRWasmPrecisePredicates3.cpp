#include "MRWasmBindings.h"

#include "MRMesh/MRPrecisePredicates3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_precise_predicates3 )
{
    // Opaque carrier: holds the float<->int coordinate converters (std::function members) that flow
    // through the precise-intersection pipeline; produced by getVectorConverters, never inspected from JS.
    emscripten::class_<CoordinateConverters>( "CoordinateConverters" );
}
