#include "MRPython/MRPython.h"
#include "MRMesh/MRContoursCut.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRUniteManyMeshes.h"
#include "MRMesh/MRId.h"

#include <pybind11/functional.h>

#include <variant>

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, EdgeTri, MR::EdgeTri )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VariableEdgeTri, MR::VariableEdgeTri, MR::EdgeTri )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, OneMeshIntersection, MR::OneMeshIntersection )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, OneMeshContour, MR::OneMeshContour )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshIntersectinosTypes, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( EdgeTri ).doc() =
        "edge from one mesh and triangle from another mesh";
    MR_PYTHON_CUSTOM_CLASS( EdgeTri ).
        def( pybind11::init<>() ).
        def_readwrite( "edge", &MR::EdgeTri::edge ).
        def_readwrite( "tri", &MR::EdgeTri::tri );

    MR_PYTHON_CUSTOM_CLASS( VariableEdgeTri ).
        def( pybind11::init<>() ).
        def_readwrite( "isEdgeATriB", &MR::VariableEdgeTri::isEdgeATriB );

    pybind11::enum_<MR::OneMeshIntersection::VariantIndex>( m, "OneMeshIntersectionVariantType" ).
        value( "Face", MR::OneMeshIntersection::VariantIndex::Face ).
        value( "Edge", MR::OneMeshIntersection::VariantIndex::Edge ).
        value( "Vertex", MR::OneMeshIntersection::VariantIndex::Vertex );

    using OneMeshIntersectionVariant = std::variant<MR::FaceId, MR::EdgeId, MR::VertId>;
    pybind11::class_<OneMeshIntersectionVariant>( m, "OneMeshIntersectionVariant" ).
        def( pybind11::init<>() ).
        def( "index", [] ( const OneMeshIntersectionVariant& self ) { return self.index(); } ).
        def( "getFace", [] ( const OneMeshIntersectionVariant& self ) { return std::get<MR::FaceId>( self ); } ).
        def( "getEdge", [] ( const OneMeshIntersectionVariant& self ) { return std::get<MR::EdgeId>( self ); } ).
        def( "getVert", [] ( const OneMeshIntersectionVariant& self ) { return std::get<MR::VertId>( self ); } );

    MR_PYTHON_CUSTOM_CLASS( OneMeshIntersection ).doc() =
        "Simple point on mesh, represented by primitive id and coordinate in mesh space";
    MR_PYTHON_CUSTOM_CLASS( OneMeshIntersection ).
        def( pybind11::init<>() ).
        def_readwrite( "primitiveId", &MR::OneMeshIntersection::primitiveId ).
        def_readwrite( "coordinate", &MR::OneMeshIntersection::coordinate );

    pybind11::class_<MR::CoordinateConverters>( m, "CoordinateConverters", "this struct contains coordinate converters float-int-float" ).
        def( pybind11::init<>() );

    m.def( "getVectorConverters", &MR::getVectorConverters, pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr,
        "creates simple converters from Vector3f to Vector3i and back in mesh parts area range\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation" );

} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorEdgeTri, MR::EdgeTri )

MR_ADD_PYTHON_VEC( mrmeshpy, ContinuousContour, MR::VariableEdgeTri )

MR_ADD_PYTHON_VEC( mrmeshpy, ContinuousContours, MR::ContinuousContour )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorOneMeshIntersection, MR::OneMeshIntersection )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshIntersectinosTypes2, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( OneMeshContour ).doc() =
        "One contour on mesh";
    MR_PYTHON_CUSTOM_CLASS( OneMeshContour ).
        def( pybind11::init<>() ).
        def_readwrite( "intersections", &MR::OneMeshContour::intersections ).
        def_readwrite( "closed", &MR::OneMeshContour::closed );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, OneMeshContours, MR::OneMeshContour )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, BooleanExposing, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::PreciseCollisionResult>( m, "PreciseCollisionResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "edgesAtrisB", &MR::PreciseCollisionResult::edgesAtrisB, "each edge is directed to have its origin inside and its destination outside of the other mesh" ).
        def_readwrite( "edgesBtrisA", &MR::PreciseCollisionResult::edgesBtrisA, "each edge is directed to have its origin inside and its destination outside of the other mesh" );

    m.def( "findCollidingEdgeTrisPrecise", [] ( const MR::MeshPart& pA, const MR::MeshPart& pB, MR::CoordinateConverters conv, const MR::AffineXf3f* xf, bool any )
    {
        return MR::findCollidingEdgeTrisPrecise( pA, pB, conv.toInt, xf, any );
    },
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "conv" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "anyIntersection" ) = false,
        "finds all pairs of colliding edges from one mesh and triangle from another mesh\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tanyIntersection - if true then the function returns as fast as it finds any intersection" );

    m.def( "orderIntersectionContours", &MR::orderIntersectionContours, pybind11::arg( "topologyA" ), pybind11::arg( "topologyB" ), pybind11::arg( "intersections" ),
        "Combines individual intersections into ordered contours with the properties:\n"
        "  a. left  of contours on mesh A is inside of mesh B,\n"
        "  b. right of contours on mesh B is inside of mesh A,\n"
        "  c. each intersected edge has origin inside meshes intersection and destination outside of it" );

    m.def( "getOneMeshIntersectionContours", &MR::getOneMeshIntersectionContours,
        pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "contours" ), pybind11::arg( "getMeshAIntersections" ), pybind11::arg( "converters" ), pybind11::arg( "rigidB2A" ) = nullptr,
        "Converts ordered continuous contours of two meshes to OneMeshContours\n"
        "converters is required for better precision in case of degenerations\n"
        "note that contours should not have intersections" );

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
