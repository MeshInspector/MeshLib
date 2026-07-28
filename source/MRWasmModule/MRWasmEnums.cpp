#include "MRWasmBindings.h"

#include "MRMesh/MREnums.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_enums )
{
    emscripten::enum_<EdgeWeights>( "EdgeWeights" )
        .value( "Unit", EdgeWeights::Unit )
        .value( "Cotan", EdgeWeights::Cotan );

    emscripten::enum_<VertexMass>( "VertexMass" )
        .value( "Unit", VertexMass::Unit )
        .value( "NeiArea", VertexMass::NeiArea );

    emscripten::enum_<RememberShape>( "RememberShape" )
        .value( "Yes", RememberShape::Yes )
        .value( "No", RememberShape::No );

    emscripten::enum_<OffsetMode>( "OffsetMode" )
        .value( "Smooth", OffsetMode::Smooth )
        .value( "Standard", OffsetMode::Standard )
        .value( "Sharpening", OffsetMode::Sharpening );
}
