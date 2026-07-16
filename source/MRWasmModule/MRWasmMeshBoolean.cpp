#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRBooleanOperation.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVector.h"

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

    emscripten::function( "boolean", +[]( std::shared_ptr<Mesh> meshA, std::shared_ptr<Mesh> meshB, BooleanOperation operation )
    {
        return boolean( *meshA, *meshB, operation );
    } );

    emscripten::enum_<BooleanResultMapper::MapObject>( "BooleanMapObject" )
        .value( "A", BooleanResultMapper::MapObject::A )
        .value( "B", BooleanResultMapper::MapObject::B );

    emscripten::class_<BooleanResultMapper>( "BooleanResultMapper" )
        .constructor<>()
        .function( "mapFaces", +[]( const BooleanResultMapper& m, const FaceBitSet& oldBS, BooleanResultMapper::MapObject obj )
        {
            return m.map( oldBS, obj );
        } )
        .function( "mapVerts", +[]( const BooleanResultMapper& m, const VertBitSet& oldBS, BooleanResultMapper::MapObject obj )
        {
            return m.map( oldBS, obj );
        } )
        .function( "newFaces", +[]( const BooleanResultMapper& m ) { return m.newFaces(); } )
        .function( "filteredOldFaceBitSet", +[]( const BooleanResultMapper& m, const FaceBitSet& oldBS, BooleanResultMapper::MapObject obj )
        {
            return m.filteredOldFaceBitSet( oldBS, obj );
        } )
        .function( "getNew2OldFaceMap", +[]( const BooleanResultMapper& m, BooleanResultMapper::MapObject obj )
        {
            return m.getNew2OldFaceMap( obj );
        } );

    emscripten::function( "boolean", +[]( std::shared_ptr<Mesh> meshA, std::shared_ptr<Mesh> meshB, BooleanOperation operation, BooleanResultMapper& mapper )
    {
        return boolean( *meshA, *meshB, operation, nullptr, &mapper );
    } );
}
