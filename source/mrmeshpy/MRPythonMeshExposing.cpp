#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRCube.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRBoolean.h"
#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshNormals.h"

using namespace MR;

Mesh pythonGetSelectedMesh()
{
    auto selected = MR::getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return {};
    if ( !selected[0] || !selected[0]->mesh() )
        return {};
    return *selected[0]->mesh();
}

void pythonSetMeshToSelected( Mesh mesh )
{
    auto selected = MR::getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return;
    if ( !selected[0] )
        return;
    selected[0]->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    selected[0]->setDirtyFlags( DIRTY_ALL );
}

MR_ADD_PYTHON_FUNCTION( mrmeshpy, getSelectedMesh, pythonGetSelectedMesh, "copy selected mesh from scene tree" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, setMeshToSelected, pythonSetMeshToSelected, "add mesh to scene tree and select it" )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshTopology, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshTopology>( m, "MeshTopology" ).
        def( pybind11::init<>() ).
        def( "getValidFaces", &MR::MeshTopology::getValidFaces, pybind11::return_value_policy::copy, "returns cached set of all valid faces" ).
        def( "getValidVerts", &MR::MeshTopology::getValidVerts, pybind11::return_value_policy::copy, "returns cached set of all valid vertices" ).
        def( "org", &MR::MeshTopology::org, pybind11::arg( "he" ), "returns origin vertex of half-edge" ).
        def( "dest", &MR::MeshTopology::dest, pybind11::arg( "he" ), "returns destination vertex of half-edge" ).
        def( "findBoundaryFaces", &MR::MeshTopology::findBoundaryFaces, "returns all boundary faces, having at least one boundary edge" ).
        def( "findBoundaryEdges", &MR::MeshTopology::findBoundaryEdges, "returns all boundary edges, where each edge does not have valid left face" ).
        def( "findBoundaryVerts", &MR::MeshTopology::findBoundaryVerts, "returns all boundary vertices, incident to at least one boundary edge" ).
        def( "deleteFaces", &MR::MeshTopology::deleteFaces, pybind11::arg( "fs" ), "deletes multiple given faces" ).
        def( "findBoundary", &MR::MeshTopology::findBoundary, pybind11::arg( "region" ) = nullptr,
            "returns all boundary loops, where each edge has region face to the right and does not have valid or in-region left face;\n"
            "unlike findRegionBoundary this method returns loops in opposite orientation" ).
        def( "findHoleRepresentiveEdges", &MR::MeshTopology::findHoleRepresentiveEdges, "returns one edge with no valid left face for every boundary in the mesh" ).
        def( "getTriVerts", ( void( MR::MeshTopology::* )( FaceId, VertId&, VertId&, VertId& )const )& MR::MeshTopology::getTriVerts,
            pybind11::arg("f"), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ), 
            "gets 3 vertices of given triangular face;\n"
            "the vertices are returned in counter-clockwise order if look from mesh outside" ).
        def( pybind11::self == pybind11::self, "compare that two topologies are exactly the same" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Vector, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::VertCoords>( m, "VertCoords" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::VertCoords::vec_ );

    pybind11::class_<MR::FaceMap>( m, "FaceMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::FaceMap::vec_ );

    pybind11::class_<MR::VertMap>( m, "VertMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::VertMap::vec_ );

    pybind11::class_<MR::EdgeMap>( m, "EdgeMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::EdgeMap::vec_ );

    pybind11::class_<MR::Vector<float, VertId>>( m, "VectorFloatByVert" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::Vector<float, VertId>::vec_ );

    pybind11::class_<MR::FaceNormals>( m, "FaceNormals" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::FaceNormals::vec_ );

    pybind11::class_<MR::Vector<Vector2f, VertId>>( m, "VertCoords2" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::Vector<Vector2f, VertId>::vec_ );

    pybind11::class_<MR::Vector<Color, VertId>>( m, "VertColorMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::Vector<Color, VertId>::vec_ );
} )

MR::MeshTopology topologyFromTriangles( const Triangulation& t, const MeshBuilder::BuildSettings& s )
{
    return MR::MeshBuilder::fromTriangles( t, s );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilder, []( pybind11::module_& m )
{
    pybind11::class_<MR::ThreeVertIds>( m, "ThreeVertIds" ).
        def( pybind11::init( [] ( MR::VertId v0, MR::VertId v1, MR::VertId v2 )->MR::ThreeVertIds
    {
        return { v0, v1, v2 };
    } ) );

    pybind11::class_<MR::Triangulation>( m, "Triangulation" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &Triangulation::vec_ );

    pybind11::class_<MR::MeshBuilder::BuildSettings>( m, "MeshBuilderSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "region", &MR::MeshBuilder::BuildSettings::region, "if region is given then on input it contains the faces to be added, and on output the faces failed to be added" ).
        def_readwrite( "shiftFaceId", &MR::MeshBuilder::BuildSettings::shiftFaceId, "this value to be added to every faceId before its inclusion in the topology" ).
        def_readwrite( "allowNonManifoldEdge", &MR::MeshBuilder::BuildSettings::allowNonManifoldEdge, "whether to permit non-manifold edges in the resulting topology" );

    m.def( "topologyFromTriangles",
        ( MeshTopology( * )( const Triangulation&, const MeshBuilder::BuildSettings& ) )& topologyFromTriangles,
        pybind11::arg( "triangulation" ), pybind11::arg( "settings" ) = MeshBuilder::BuildSettings{},
        "construct mesh topology from a set of triangles with given ids;\n"
        "if skippedTris is given then it receives all input triangles not added in the resulting topology" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorThreeVertIds, MR::ThreeVertIds )

MR::Mesh pythonCopyMeshFunction( const MR::Mesh& mesh )
{
    return mesh;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Mesh, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Mesh>( m, "Mesh" ).
        def( pybind11::init<>() ).
        def( "computeBoundingBox", ( Box3f( MR::Mesh::* )( const FaceBitSet*, const AffineXf3f* ) const )& MR::Mesh::computeBoundingBox,
            pybind11::arg( "region" ) = nullptr, pybind11::arg( "toWorld" ) = nullptr,
            "passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them\n"
            "if toWorld transformation is given then returns minimal bounding box in world space" ).
        def( "getBoundingBox", &MR::Mesh::getBoundingBox,
            "returns the bounding box containing all valid vertices (implemented via getAABBTree())\n"
            "this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision" ).
        def( "area", ( double( MR::Mesh::* )( const FaceBitSet* fs )const )& MR::Mesh::area, pybind11::arg( "fs" ) = nullptr, "this version returns the area of whole mesh if argument is nullptr" ).
        def( "volume", &MR::Mesh::volume, pybind11::arg( "region" ) = nullptr,
            "returns volume of closed mesh region, if region is not closed DBL_MAX is returned\n"
            "if region is nullptr - whole mesh is region" ).
        def( "pack", &MR::Mesh::pack, pybind11::arg( "outFmap" ) = nullptr, pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr, pybind11::arg( "rearrangeTriangles" ) = false,
            "tightly packs all arrays eliminating lone edges and invalid face, verts and points,\n"
            "optionally returns mappings: old.id -> new.id" ).
        def( "discreteMeanCurvature", &MR::Mesh::discreteMeanCurvature, pybind11::arg( "v" ),
            "computes discrete mean curvature in given vertex measures in length^-1;\n"
            "0 for planar regions, positive for convex surface, negative for concave surface" ).
        def_readwrite( "topology", &MR::Mesh::topology ).
        def_readwrite( "points", &MR::Mesh::points ).
        def( "triPoint", ( MR::Vector3f( MR::Mesh::* )( const MR::MeshTriPoint& )const )& MR::Mesh::triPoint, pybind11::arg( "p" ), "returns interpolated coordinates of given point" ).
        def( "edgePoint", ( MR::Vector3f( MR::Mesh::* )( const MR::MeshEdgePoint& )const )& MR::Mesh::edgePoint, pybind11::arg( "ep" ), "returns a point on the edge: origin point for f=0 and destination point for f=1" ).
        def( "invalidateCaches", &MR::Mesh::invalidateCaches, "Invalidates caches (e.g. aabb-tree) after a change in mesh geometry or topology" ).
        def( "transform", ( void( MR::Mesh::* ) ( const AffineXf3f& ) )& MR::Mesh::transform, pybind11::arg( "xf" ), "applies given transformation to all valid mesh vertices" ).
        def( pybind11::self == pybind11::self, "compare that two meshes are exactly the same" );

    m.def( "copyMesh", &pythonCopyMeshFunction, pybind11::arg( "mesh" ), "returns copy of input mesh" );
} )

MR_ADD_PYTHON_EXPECTED( mrmeshpy, ExpectedMesh, MR::Mesh, std::string )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPart, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshPart>( m, "MeshPart", "stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)" ).
        def( pybind11::init<const Mesh&, const FaceBitSet*>(), pybind11::arg( "mesh" ), pybind11::arg( "region" ) = nullptr ).
        def_readwrite( "region", &MR::MeshPart::region, "nullptr here means whole mesh" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVertBitSet, MR::VertBitSet )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceBitSet, MR::FaceBitSet )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, RegionBoundary, [] ( pybind11::module_& m )
{
    m.def( "getIncidentVerts", ( MR::VertBitSet( * )( const MR::MeshTopology&, const MR::FaceBitSet& ) )& MR::getIncidentVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices incident to given faces" );
    m.def( "getInnerVerts", ( MR::VertBitSet( * )( const MR::MeshTopology&, const MR::FaceBitSet& ) )& MR::getInnerVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices with all their faces in given set" );

    m.def( "getIncidentFaces", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::VertBitSet& ) )& MR::getIncidentFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces incident to given vertices" );
    m.def( "getInnerFaces", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::VertBitSet& ) )& MR::getInnerFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces with all their vertices in given set" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshComponents, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::MeshComponents::FaceIncidence>( m, "FaceIncidence", "Face incidence type" ).
        value( "PerEdge", MR::MeshComponents::FaceIncidence::PerEdge, "face can have neighbor only via edge" ).
        value( "PerVertex", MR::MeshComponents::FaceIncidence::PerVertex, "face can have neighbor via vertex" );

    m.def( "getAllComponentsVerts", &MR::MeshComponents::getAllComponentsVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "gets all connected components of mesh part" );

    m.def( "getAllComponents", &MR::MeshComponents::getAllComponents,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "gets all connected components of mesh part" );

    m.def( "getComponent", &MR::MeshComponents::getComponent,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "faceId" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "returns one connected component containing given face,\n"
        "not effective to call more than once, if several components are needed use getAllComponents" );

    m.def( "getComponentVerts", &MR::MeshComponents::getComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "vertId" ),
        pybind11::arg( "region" ) = nullptr,
        "returns one connected component containing given vertex,\n"
        "not effective to call more than once, if several components are needed use getAllComponentsVerts" );

    m.def( "getLargestComponent", &MR::MeshComponents::getLargestComponent,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "returns largest by surface area component" );

    m.def( "getLargestComponentVerts", &MR::MeshComponents::getLargestComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "returns largest by number of elements component" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorMesh, MR::Mesh )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FillHole, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::FillHoleParams::MultipleEdgesResolveMode>( m, "FillHoleParamsMultipleEdgesResolveMode" ).
        value( "None", MR::FillHoleParams::MultipleEdgesResolveMode::None, "do not avoid multiple edges" ).
        value( "Simple", MR::FillHoleParams::MultipleEdgesResolveMode::Simple, "avoid creating edges that already exist in topology (default)" ).
        value( "Strong", MR::FillHoleParams::MultipleEdgesResolveMode::Strong,
            "makes additional efforts to avoid creating multiple edges,\n"
            "in some rare cases it is not possible (cases with extremely bad topology),\n"
            "if you faced one try to use duplicateMultiHoleVertices before fillHole" );

    pybind11::class_<MR::FillHoleMetric>( m, "FillHoleMetric", "This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions" ).
        def( pybind11::init<>() );

    m.def( "getPlaneFillMetric", &MR::getPlaneFillMetric, pybind11::arg( "mesh" ), pybind11::arg( "e" ),
        "Same as getCircumscribedFillMetric, but with extra penalty for the triangles having\n"
        "normals looking in the opposite side of plane containing left of (e)." );
    
    m.def( "getEdgeLengthFillMetric", &MR::getEdgeLengthFillMetric, pybind11::arg( "mesh" ),
        "Simple metric minimizing the sum of all edge lengths" );
    
    m.def( "getEdgeLengthStitchMetric", &MR::getEdgeLengthStitchMetric, pybind11::arg( "mesh" ),
        "Forbids connecting vertices from the same hole\n"
        "Simple metric minimizing edge length" );

    m.def( "getCircumscribedMetric", &MR::getCircumscribedMetric, pybind11::arg( "mesh" ),
        "This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.\n"
        "It is rather fast to calculate, and it results in typically good triangulations." );

    pybind11::class_<MR::FillHoleParams>( m, "FillHoleParams", "Structure has some options to control fillHole" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &MR::FillHoleParams::metric, "Specifies triangulation metric\n""default for fillHole: getCircumscribedFillMetric" ).
        def_readwrite( "multipleEdgesResolveMode", &MR::FillHoleParams::multipleEdgesResolveMode ).
        def_readwrite( "makeDegenerateBand", &MR::FillHoleParams::makeDegenerateBand,
            "If true creates degenerate faces band around hole to have sharp angle visualization\n"
            "warning: This flag bad for result topology, most likely you do not need it" ).
        def_readwrite( "maxPolygonSubdivisions", &MR::FillHoleParams::maxPolygonSubdivisions, "The maximum number of polygon subdivisions on a triangle and two smaller polygons,\n""must be 2 or larger" ).
        def_readwrite( "outNewFaces", &MR::FillHoleParams::outNewFaces, "If not nullptr accumulate new faces" );

    pybind11::class_<MR::StitchHolesParams>( m, "StitchHolesParams", "Structure has some options to control buildCylinderBetweenTwoHoles" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &StitchHolesParams::metric,
            "Specifies triangulation metric\n"
            "default for buildCylinderBetweenTwoHoles: getComplexStitchMetric").
        def_readwrite( "outNewFaces", &StitchHolesParams::outNewFaces, "If not nullptr accumulate new faces" );

    m.def( "fillHole", &MR::fillHole,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "params" ) = MR::FillHoleParams{},
        "Fills given hole represented by one of its edges (having no valid left face),\n"
        "uses fillHoleTrivially if cannot fill hole without multiple edges,\n"
        "default metric: CircumscribedFillMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents hole\n"
        "\tparams - parameters of hole filling" );

    m.def( "buildCylinderBetweenTwoHoles", ( void ( * )( Mesh&, EdgeId, EdgeId, const StitchHolesParams& ) )& MR::buildCylinderBetweenTwoHoles,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "params" ) = MR::StitchHolesParams{},
        "Build cylindrical patch to fill space between two holes represented by one of their edges each,\n"
        "default metric: ComplexStitchMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents 1st hole\n"
        "\tb - EdgeId which represents 2nd hole\n"
        "\tparams - parameters of holes stitching" );

    m.def( "buildCylinderBetweenTwoHoles", ( bool ( * )( Mesh&, const StitchHolesParams& ) )& MR::buildCylinderBetweenTwoHoles,
       pybind11::arg( "mesh" ), pybind11::arg( "params" ) = MR::StitchHolesParams{},
       "this version finds holes in the mesh by itself and returns false if they are not found" );
} )

Mesh pythonMergeMehses( const pybind11::list& meshes )
{
    Mesh res;
    for ( int i = 0; i < pybind11::len( meshes ); ++i )
        res.addPart( pybind11::cast<Mesh>( meshes[i] ) );
    return res;
}

MR::FaceBitSet getFacesByMinEdgeLength( const MR::Mesh& mesh, float minLength )
{
    using namespace MR;
    FaceBitSet resultFaces( mesh.topology.getValidFaces().size() );
    float minLengthSq = minLength * minLength;
    for ( auto ue : MR::undirectedEdges( mesh.topology ) )
    {
        if ( mesh.edgeLengthSq( ue ) > minLengthSq )
        {
            auto l = mesh.topology.left( ue );
            auto r = mesh.topology.right( ue );
            if ( l )
                resultFaces.set( l );
            if ( r )
                resultFaces.set( r );
        }
    }
    return resultFaces;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SimpleFunctions, [] ( pybind11::module_& m )
{
    m.def( "computePerVertNormals", &MR::computePerVertNormals, pybind11::arg( "mesh" ), "returns a vector with vertex normals in every element for valid mesh vertices" );
    m.def( "computePerFaceNormals", &MR::computePerFaceNormals, pybind11::arg( "mesh" ), "returns a vector with face-normal in every element for valid mesh faces" );
    m.def( "mergeMehses", &pythonMergeMehses, pybind11::arg( "meshes" ), "merge python list of meshes to one mesh" );
    m.def( "getFacesByMinEdgeLength", &getFacesByMinEdgeLength, pybind11::arg( "mesh" ), pybind11::arg( "minLength" ), "return faces with at least one edge longer than min edge length" );
    m.def( "buildBottom", &MR::buildBottom, pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "dir" ), pybind11::arg( "holeExtension" ), pybind11::arg( "outNewFaces" ) = nullptr,
        "adds cylindrical extension of given hole represented by one of its edges (having no valid left face)\n"
        "by adding new vertices located in lowest point of the hole -dir*holeExtension and 2 * number_of_hole_edge triangles;\n"
        "return: the edge of new hole opposite to input edge (a)" );

    m.def( "makeCube", &MR::makeCube, pybind11::arg( "size" ) = MR::Vector3f::diagonal( 1 ), pybind11::arg( "base" ) = MR::Vector3f::diagonal( -0.5f ),
        "Base is \"lower\" corner of the cube coordinates" );

    m.def( "makeTorus", &MR::makeTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "Z is symmetry axis of this torus\n"
        "points - optional out points of main circle" );

    m.def( "makeOuterHalfTorus", &MR::makeOuterHalfTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus without inner half faces\n"
        "main application - testing fillHole and Stitch" );

    m.def( "makeTorusWithUndercut", &MR::makeTorusWithUndercut,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with inner protruding half as undercut\n"
        "main application - testing fixUndercuts" );

    m.def( "makeTorusWithSpikes", &MR::makeTorusWithSpikes,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with some handed-up points\n"
        "main application - testing fixSpikes and Relax" );

    m.def( "makeTorusWithComponents", &MR::makeTorusWithComponents,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );

    m.def( "makeTorusWithSelfIntersections", &MR::makeTorusWithSelfIntersections,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );


    pybind11::class_<MR::FaceFace>( m, "FaceFace" ).
        def( pybind11::init<>() ).
        def( pybind11::init<FaceId, FaceId>(), pybind11::arg( "a" ), pybind11::arg( "b" ) ).
        def_readwrite( "aFace", &MR::FaceFace::aFace ).
        def_readwrite( "bFace", &MR::FaceFace::bFace );

    m.def( "findSelfCollidingTriangles", &MR::findSelfCollidingTriangles, pybind11::arg( "mp" ), "finds all pairs of colliding triangles from one mesh or a region" );

    m.def( "findSelfCollidingTrianglesBS", &MR::findSelfCollidingTrianglesBS, pybind11::arg( "mp" ), "finds union of all self-intersecting faces" );

    m.def( "findCollidingTriangles", &MR::findCollidingTriangles, 
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "firstIntersectionOnly" ) = false, 
        "finds all pairs of colliding triangles from two meshes or two mesh regions\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tfirstIntersectionOnly - if true then the function returns at most one pair of intersecting triangles and returns faster" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceFace, MR::FaceFace )
