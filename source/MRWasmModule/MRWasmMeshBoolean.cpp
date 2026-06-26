#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRBooleanOperation.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_boolean )
{
    emscripten::enum_<BooleanOperation>( "BooleanOperation" )
        .value( "InsideA", BooleanOperation::InsideA )
        .value( "InsideB", BooleanOperation::InsideB )
        .value( "OutsideA", BooleanOperation::OutsideA )
        .value( "OutsideB", BooleanOperation::OutsideB )
        .value( "Union", BooleanOperation::Union )
        .value( "Intersection", BooleanOperation::Intersection )
        .value( "DifferenceBA", BooleanOperation::DifferenceBA )
        .value( "DifferenceAB", BooleanOperation::DifferenceAB );

    emscripten::class_<BooleanResult>( "BooleanResult" )
        .property( "errorString", &BooleanResult::errorString )
        .property( "mesh", +[]( const BooleanResult& r ) { return std::make_shared<Mesh>( r.mesh ); } )
        .function( "valid", &BooleanResult::valid );

    emscripten::function( "boolean", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b, BooleanOperation op )
    {
        return boolean( *a, *b, op );
    } );
}
