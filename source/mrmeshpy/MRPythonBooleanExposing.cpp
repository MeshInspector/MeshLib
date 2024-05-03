#include "MRMesh/MRPython.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRUniteManyMeshes.h"
#include <pybind11/functional.h>

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
    
    pybind11::class_<MR::BooleanResultMapper::Maps>( m, "BooleanResMaps" ).
        def( pybind11::init<>() ).
        def_readwrite( "cut2origin", &MR::BooleanResultMapper::Maps::cut2origin,
            "\"after cut\" faces to \"origin\" faces\n"
            "this map is not 1-1, but N-1" ).
        def_readwrite( "cut2newFaces", &MR::BooleanResultMapper::Maps::cut2newFaces, "\"after cut\" faces to \"after stitch\" faces( 1 - 1 )" ).
        def_readwrite( "old2newEdges", &MR::BooleanResultMapper::Maps::old2newEdges, "\"origin\" edges to \"after stitch\" edges (1-1)" ).
        def_readwrite( "old2newVerts", &MR::BooleanResultMapper::Maps::old2newVerts, "\"origin\" vertices to \"after stitch\" vertices (1-1)" ).
        def_readwrite( "identity", &MR::BooleanResultMapper::Maps::identity, "old topology indexes are valid if true" );

    pybind11::class_<MR::BooleanResultMapper>( m, "BooleanResultMapper", "This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MR::boolean to result mesh topology primitives" ).
        def( pybind11::init<>() ).
        def( "map", ( MR::VertBitSet( MR::BooleanResultMapper::* )( const MR::VertBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns vertices bitset of result mesh corresponding input one" ).
        def( "map", ( MR::EdgeBitSet( MR::BooleanResultMapper::* )( const MR::EdgeBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns edges bitset of result mesh corresponding input one" ).
        def( "map", ( MR::FaceBitSet( MR::BooleanResultMapper::* )( const MR::FaceBitSet&, MR::BooleanResultMapper::MapObject )const )& MR::BooleanResultMapper::map,
            pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "Returns faces bitset of result mesh corresponding input one" ).
        def( "filteredOldFaceBitSet", &MR::BooleanResultMapper::filteredOldFaceBitSet, pybind11::arg( "oldBS" ), pybind11::arg( "obj" ), "returns updated oldBS leaving only faces that has corresponding ones in result mesh" ).
        def( "getMaps", [] ( MR::BooleanResultMapper& mapper, MR::BooleanResultMapper::MapObject obj )->MR::BooleanResultMapper::Maps& { return mapper.maps[int( obj )]; } );

    m.def( "boolean", ( MR::BooleanResult( * )( const MR::Mesh&, const MR::Mesh&, MR::BooleanOperation, const MR::AffineXf3f*, MR::BooleanResultMapper*, MR::ProgressCallback ) )MR::boolean,
        pybind11::arg("meshA"), pybind11::arg( "meshB" ), pybind11::arg( "operation" ),
        pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "mapper" ) = nullptr, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Makes new mesh - result of boolean operation on mesh `A` and mesh `B`\n"
        "\tmeshA - Input mesh `A`\n"
        "\tmeshB - Input mesh `B`\n"
        "\toperation - CSG operation to perform\n"
        "\trigidB2A - Transform from mesh `B` space to mesh `A` space\n"
        "\tmapper - Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology\n\n"
        "note: Input meshes should have no self-intersections in intersecting zone\n"
        "note: If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)" );

    pybind11::enum_<MR::NestedComponenetsMode>( m, "NestedComponenetsMode", "Mode of processing components" ).
        value( "Remove", MR::NestedComponenetsMode::Remove, "Default: separate nested meshes and remove them, just like union operation should do, use this if input meshes are single component" ).
        value( "Merge", MR::NestedComponenetsMode::Merge, "merge nested meshes, useful if input meshes are components of single object" ).
        value( "Union", MR::NestedComponenetsMode::Union, "does not separate components and call union for all input meshes, works slower than Remove and Merge method but returns valid result if input meshes has multiple components" );

    pybind11::class_<MR::UniteManyMeshesParams>( m, "UniteManyMeshesParams", "Parameters structure for uniteManyMeshes function" ).
        def( pybind11::init<>() ).
        def_readwrite( "useRandomShifts", &MR::UniteManyMeshesParams::useRandomShifts, "Apply random shift to each mesh, to prevent degenerations on coincident surfaces" ).
        def_readwrite( "fixDegenerations", &MR::UniteManyMeshesParams::fixDegenerations,
            "Try fix degenerations after each boolean step, to prevent boolean failure due to high amount of degenerated faces\n"
            "useful on meshes with many coincident surfaces \n"
            "(useRandomShifts used for same issue)" ).
        def_readwrite( "maxAllowedError", &MR::UniteManyMeshesParams::maxAllowedError,
            "Max allowed random shifts in each direction, and max allowed deviation after degeneration fixing\n"
            "not used if both flags (useRandomShifts,fixDegenerations) are false" ).
        def_readwrite( "randomShiftsSeed", &MR::UniteManyMeshesParams::randomShiftsSeed, "Seed that is used for random shifts" ).
        def_readwrite( "nestedComponentsMode", &MR::UniteManyMeshesParams::nestedComponentsMode,
            "By default function separate nested meshes and remove them, just like union operation should do\n"
            "read comment of NestedComponenetsMode enum for more information" ).
        def_readwrite( "newFaces", &MR::UniteManyMeshesParams::newFaces, "If set, the bitset will store new faces created by boolean operations" );

    m.def( "uniteManyMeshes", MR::decorateExpected( &MR::uniteManyMeshes ), pybind11::arg( "meshes" ), pybind11::arg_v( "params", MR::UniteManyMeshesParams(), "UniteManyMeshesParams()" ),
        "Computes the surface of objects' union each of which is defined by its own surface mesh\n"
        "- merge non intersecting meshes first\n"
        "- unite merged groups" );
} )
