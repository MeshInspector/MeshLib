#include "MRMesh/MRPython.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMeshBoolean.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, BooleanExposing, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::BooleanOperation>( m, "BooleanOperation", "Enum class of available CSG operations" ).
        value( "InsideA", MR::BooleanOperation::InsideA, "Part of mesh `A` that is inside of mesh `B`" ).
        value( "InsideB", MR::BooleanOperation::InsideB, "Part of mesh `B` that is inside of mesh `A`" ).
        value( "OutsideA", MR::BooleanOperation::OutsideA, "Part of mesh `A` that is outside of mesh `B`" ).
        value( "OutsideB", MR::BooleanOperation::OutsideB, "Part of mesh `B` that is outside of mesh `A`" ).
        value( "Union", MR::BooleanOperation::Union, "Union surface of two meshes (outside parts)" ).
        value( "Intersection", MR::BooleanOperation::Intersection, "Intersection surface of two meshes (inside parts)" ).
        value( "DifferenceAB", MR::BooleanOperation::DifferenceAB, "Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)" ).
        value( "DifferenceBA", MR::BooleanOperation::DifferenceBA, "Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)" );
    
    
    pybind11::class_<MR::BooleanResult>( m, "BooleanResult", "This structure store result mesh of MR::boolean or some error info" ).
        def( pybind11::init<>() ).
        def_readwrite( "mesh", &MR::BooleanResult::mesh, "Result mesh of boolean operation, if error occurred it would be empty" ).
        def_readwrite( "meshABadContourFaces", &MR::BooleanResult::meshABadContourFaces, "If input contours have intersections, this face bit set presents faces of mesh `A` on which contours intersect" ).
        def_readwrite( "meshBBadContourFaces", &MR::BooleanResult::meshBBadContourFaces, "If input contours have intersections, this face bit set presents faces of mesh `B` on which contours intersect" ).
        def_readwrite( "errorString", &MR::BooleanResult::errorString, "Holds error message, empty if boolean succeed" ).
        def( "valid", &MR::BooleanResult::valid, "Returns true if boolean succeed, false otherwise" );
    
    pybind11::enum_<MR::BooleanResultMapper::MapObject>( m, "BooleanResMapObj", "Input object index enum" ).
        value( "A", MR::BooleanResultMapper::MapObject::A ).
        value( "B", MR::BooleanResultMapper::MapObject::B );
    
    pybind11::class_<MR::BooleanResultMapper>( m, "BooleanResultMapper", "This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MR::boolean to result mesh topology primitives" ).
        def( pybind11::init<>() ).
        def( "map", ( MR::VertBitSet( MR::BooleanResultMapper::* )( const MR::VertBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns vertices bitset of result mesh corresponding input one" ).
        def( "map", ( MR::EdgeBitSet( MR::BooleanResultMapper::* )( const MR::EdgeBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns edges bitset of result mesh corresponding input one" ).
        def( "map", ( MR::FaceBitSet( MR::BooleanResultMapper::* )( const MR::FaceBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns faces bitset of result mesh corresponding input one" );

    m.def( "boolean", MR::boolean, 
        pybind11::arg("meshA"), pybind11::arg( "meshB" ), pybind11::arg( "operation" ),
        pybind11::arg("rigidB2A") = nullptr, pybind11::arg( "mapper" ) = nullptr,
        "Makes new mesh - result of boolean operation on mesh `A` and mesh `B`\n"
        "\tmeshA Input mesh `A`\n"
        "\tmeshB Input mesh `B`\n"
        "\toperation CSG operation to perform\n"
        "\trigidB2A Transform from mesh `B` space to mesh `A` space\n"
        "\tmapper Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology\n\n"
        "note: Input meshes should have no self-intersections in intersecting zone\n"
        "note: If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)" );
} )
