#include "MRMesh/MRPython.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMeshBoolean.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, BooleanExposing, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::BooleanOperation>( m, "BooleanOperation" ).
        value( "InsideA", MR::BooleanOperation::InsideA ).
        value( "InsideB", MR::BooleanOperation::InsideB ).
        value( "OutsideA", MR::BooleanOperation::OutsideA ).
        value( "OutsideB", MR::BooleanOperation::OutsideB ).
        value( "Union", MR::BooleanOperation::Union ).
        value( "Intersection", MR::BooleanOperation::Intersection ).
        value( "DifferenceAB", MR::BooleanOperation::DifferenceAB ).
        value( "DifferenceBA", MR::BooleanOperation::DifferenceBA );
    
    
    pybind11::class_<MR::BooleanResult>( m, "BooleanResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "mesh", &MR::BooleanResult::mesh ).
        def_readwrite( "meshABadContourFaces", &MR::BooleanResult::meshABadContourFaces ).
        def_readwrite( "meshBBadContourFaces", &MR::BooleanResult::meshBBadContourFaces ).
        def_readwrite( "errorString", &MR::BooleanResult::errorString ).
        def( "valid", &MR::BooleanResult::valid );
    
    pybind11::enum_<MR::BooleanResultMapper::MapObject>( m, "BooleanResMapObj" ).
        value( "A", MR::BooleanResultMapper::MapObject::A ).
        value( "B", MR::BooleanResultMapper::MapObject::B );
    
    pybind11::class_<MR::BooleanResultMapper>( m, "BooleanResultMapper" ).
        def( pybind11::init<>() ).
        def( "map", ( MR::VertBitSet( MR::BooleanResultMapper::* )( const MR::VertBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map ).
        def( "map", ( MR::EdgeBitSet( MR::BooleanResultMapper::* )( const MR::EdgeBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map ).
        def( "map", ( MR::FaceBitSet( MR::BooleanResultMapper::* )( const MR::FaceBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map );

    m.def( "boolean", MR::boolean, "performs boolean operation on meshes" );
} )
