#include "MRMesh/MRPython.h"
#include "MRMesh/MRAffineXf3.h"
#include "MREAlgorithms/MREMeshBoolean.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, BooleanExposing, [] ( pybind11::module_& m )
{
    pybind11::enum_<MRE::BooleanOperation>( m, "BooleanOperation" ).
        value( "InsideA", MRE::BooleanOperation::InsideA ).
        value( "InsideB", MRE::BooleanOperation::InsideB ).
        value( "OutsideA", MRE::BooleanOperation::OutsideA ).
        value( "OutsideB", MRE::BooleanOperation::OutsideB ).
        value( "Union", MRE::BooleanOperation::Union ).
        value( "Intersection", MRE::BooleanOperation::Intersection ).
        value( "DifferenceAB", MRE::BooleanOperation::DifferenceAB ).
        value( "DifferenceBA", MRE::BooleanOperation::DifferenceBA );
    
    
    pybind11::class_<MRE::BooleanResult>( m, "BooleanResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "mesh", &MRE::BooleanResult::mesh ).
        def_readwrite( "meshABadContourFaces", &MRE::BooleanResult::meshABadContourFaces ).
        def_readwrite( "meshBBadContourFaces", &MRE::BooleanResult::meshBBadContourFaces ).
        def_readwrite( "errorString", &MRE::BooleanResult::errorString ).
        def( "valid", &MRE::BooleanResult::valid );
    
    pybind11::enum_<MRE::BooleanResultMapper::MapObject>( m, "BooleanResMapObj" ).
        value( "A", MRE::BooleanResultMapper::MapObject::A ).
        value( "B", MRE::BooleanResultMapper::MapObject::B );
    
    pybind11::class_<MRE::BooleanResultMapper>( m, "BooleanResultMapper" ).
        def( pybind11::init<>() ).
        def( "map", ( MR::VertBitSet( MRE::BooleanResultMapper::* )( const MR::VertBitSet&, MRE::BooleanResultMapper::MapObject )const )& MRE::BooleanResultMapper::map ).
        def( "map", ( MR::EdgeBitSet( MRE::BooleanResultMapper::* )( const MR::EdgeBitSet&, MRE::BooleanResultMapper::MapObject )const )& MRE::BooleanResultMapper::map ).
        def( "map", ( MR::FaceBitSet( MRE::BooleanResultMapper::* )( const MR::FaceBitSet&, MRE::BooleanResultMapper::MapObject )const )& MRE::BooleanResultMapper::map );

    m.def( "boolean", MRE::boolean, "performs boolean operation on meshes" );
} )
