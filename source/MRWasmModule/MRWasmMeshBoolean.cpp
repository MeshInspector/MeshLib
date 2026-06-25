#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRBooleanOperation.h"

#include <emscripten/bind.h>

#include <stdexcept>
#include <utility>

using namespace emscripten;

namespace
{

enum class JsBooleanOp { Union, Intersection, DifferenceAB, DifferenceBA };

MR::BooleanOperation toCoreOp( JsBooleanOp op )
{
    switch ( op )
    {
    case JsBooleanOp::Union:        return MR::BooleanOperation::Union;
    case JsBooleanOp::Intersection: return MR::BooleanOperation::Intersection;
    case JsBooleanOp::DifferenceAB: return MR::BooleanOperation::DifferenceAB;
    case JsBooleanOp::DifferenceBA: return MR::BooleanOperation::DifferenceBA;
    }
    throw std::runtime_error( "boolean: unknown operation" );
}

MR::Mesh booleanOp( const MR::Mesh& a, const MR::Mesh& b, JsBooleanOp op )
{
    return guarded( [&]() -> MR::Mesh
    {
        MR::BooleanResult res = MR::boolean( a, b, toCoreOp( op ) );
        if ( !res.valid() )
            throw std::runtime_error( res.errorString.empty() ? "boolean operation failed" : res.errorString );
        return std::move( res.mesh );
    } );
}

}

EMSCRIPTEN_BINDINGS( meshlib_boolean )
{
    enum_<JsBooleanOp>( "BooleanOp" )
        .value( "Union", JsBooleanOp::Union )
        .value( "Intersection", JsBooleanOp::Intersection )
        .value( "DifferenceAB", JsBooleanOp::DifferenceAB )
        .value( "DifferenceBA", JsBooleanOp::DifferenceBA );

    function( "boolean", &booleanOp );
}
