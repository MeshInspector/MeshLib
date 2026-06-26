#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRBooleanOperation.h"

#include <emscripten/bind.h>

#include <memory>

using namespace emscripten;

namespace
{

MR::BooleanResult booleanWrap( std::shared_ptr<MR::Mesh> a, std::shared_ptr<MR::Mesh> b, MR::BooleanOperation op )
{
    return MR::boolean( *a, *b, op );
}

}

EMSCRIPTEN_BINDINGS( meshlib_boolean )
{
    enum_<MR::BooleanOperation>( "BooleanOperation" )
        .value( "InsideA", MR::BooleanOperation::InsideA )
        .value( "InsideB", MR::BooleanOperation::InsideB )
        .value( "OutsideA", MR::BooleanOperation::OutsideA )
        .value( "OutsideB", MR::BooleanOperation::OutsideB )
        .value( "Union", MR::BooleanOperation::Union )
        .value( "Intersection", MR::BooleanOperation::Intersection )
        .value( "DifferenceBA", MR::BooleanOperation::DifferenceBA )
        .value( "DifferenceAB", MR::BooleanOperation::DifferenceAB );

    class_<MR::BooleanResult>( "BooleanResult" )
        .property( "errorString", &MR::BooleanResult::errorString )
        .property( "mesh", +[]( const MR::BooleanResult& r ) { return std::make_shared<MR::Mesh>( r.mesh ); } )
        .function( "valid", &MR::BooleanResult::valid );

    function( "boolean", &booleanWrap );
}
