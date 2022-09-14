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
        def( "findBoundary", &MR::MeshTopology::findBoundary, pybind11::arg( "region" ) = nullptr,
            "returns all boundary loops, where each edge has region face to the right and does not have valid or in-region left face;\n"
            "unlike findRegionBoundary this method returns loops in opposite orientation" ).
        def( "findHoleRepresentiveEdges", &MR::MeshTopology::findHoleRepresentiveEdges, "returns one edge with no valid left face for every boundary in the mesh" ).
        def( "getTriVerts", ( void( MR::MeshTopology::* )( FaceId, VertId&, VertId&, VertId& )const )& MR::MeshTopology::getTriVerts,
            pybind11::arg("f"), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ), 
            "gets 3 vertices of given triangular face;\n"
            "the vertices are returned in counter-clockwise order if look from mesh outside" );
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
} )

MR::MeshTopology topologyFromTriangles( const Triangulation& t, const MeshBuilder::BuildSettings& s )
{
    return MR::MeshBuilder::fromTriangles( t, s );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilder, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshBuilder::BuildSettings>( m, "MeshBuilderSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "region", &MR::MeshBuilder::BuildSettings::region, "if region is given then on input it contains the faces to be added, and on output the faces failed to be added" ).
        def_readwrite( "shiftFaceId", &MR::MeshBuilder::BuildSettings::shiftFaceId, "this value to be added to every faceId before its inclusion in the topology" ).
        def_readwrite( "allowNonManifoldEdge", &MR::MeshBuilder::BuildSettings::allowNonManifoldEdge, "whether to permit non-manifold edges in the resulting topology" );

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

MR_ADD_PYTHON_VEC( mrmeshpy, vecMeshBuilderTri, MR::MeshBuilder::Triangle )

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
        def( "transform", ( void( MR::Mesh::* ) ( const AffineXf3f& ) )& MR::Mesh::transform, pybind11::arg( "xf" ), "applies given transformation to all valid mesh vertices" );

    m.def( "copyMesh", &pythonCopyMeshFunction, pybind11::arg( "mesh" ), "returns copy of input mesh" );
} )

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

void pythonSetFillHolePlaneMetric( MR::FillHoleParams& params, const Mesh& mesh, EdgeId e )
{
    params.metric = getPlaneFillMetric( mesh, e );
}

void pythonSetFillHoleEdgeLengthMetric( MR::FillHoleParams& params, const Mesh& mesh )
{
    params.metric = getEdgeLengthFillMetric( mesh );
}

void pythonSetFillHoleCircumscribedMetric( MR::FillHoleParams& params, const Mesh& mesh )
{
    params.metric = getCircumscribedMetric( mesh );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FillHole, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::FillHoleParams::MultipleEdgesResolveMode>( m, "FillHoleParamsMultipleEdgesResolveMode" ).
        value( "None", MR::FillHoleParams::MultipleEdgesResolveMode::None ).
        value( "Simple", MR::FillHoleParams::MultipleEdgesResolveMode::Simple ).
        value( "Strong", MR::FillHoleParams::MultipleEdgesResolveMode::Strong );

    pybind11::class_<MR::FillHoleParams>( m, "FillHoleParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "multipleEdgesResolveMode", &MR::FillHoleParams::multipleEdgesResolveMode ).
        def_readwrite( "makeDegenerateBand", &MR::FillHoleParams::makeDegenerateBand ).
        def_readwrite( "outNewFaces", &MR::FillHoleParams::outNewFaces );

    m.def( "set_fill_hole_metric_plane", pythonSetFillHolePlaneMetric, "set plane metric to fill hole parameters" );
    m.def( "set_fill_hole_metric_edge_length", pythonSetFillHoleEdgeLengthMetric, "set edge length metric to fill hole parameters" );
    m.def( "set_fill_hole_metric_circumscribed", pythonSetFillHoleCircumscribedMetric, "set circumscribed metric to fill hole parameters" );
    m.def( "fill_hole", MR::fillHole, "fills hole represented by edge" );
} )

std::vector<Vector3f> pythonComputePerVertNormals( const Mesh& mesh )
{
    auto res = computePerVertNormals( mesh );
    return res.vec_;
}

std::vector<Vector3f> pythonComputePerFaceNormals( const Mesh& mesh )
{
    auto res = computePerFaceNormals( mesh );
    return res.vec_;
}

std::vector<Mesh> pythonGetMeshComponents( const Mesh& mesh )
{
    auto components = MeshComponents::getAllComponents( mesh, MeshComponents::FaceIncidence::PerVertex );
    std::vector<Mesh> res( components.size() );
    for ( int i = 0; i < res.size(); ++i )
        res[i].addPartByMask( mesh, components[i] );
    return res;
}

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

MR_ADD_PYTHON_FUNCTION( mrmeshpy, getFacesByMinEdgeLength, getFacesByMinEdgeLength, "return faces with at least one edge longer than min edge length" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, merge_meshes, pythonMergeMehses, "merge python list of meshes to one mesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, compute_per_vert_normals, pythonComputePerVertNormals, "returns vector that contains normal for each valid vert" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, compute_per_face_normals, pythonComputePerFaceNormals, "returns vector that contains normal for each valid face" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, build_bottom, MR::buildBottom, "prolongs hole represented by edge to lowest point by dir, returns new EdgeId corresponding givven one" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, get_mesh_components, pythonGetMeshComponents, "find all disconnecteds components of mesh, return them as vector of meshes" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_cube, MR::makeCube, "creates simple cube mesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_torus, MR::makeTorus, "creates simple torus mesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_outer_half_test_torus, MR::makeOuterHalfTorus, "creates spetial torus without inner faces" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_undercut_test_torus, MR::makeTorusWithUndercut, "creates spetial torus with undercut" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_spikes_test_torus, MR::makeTorusWithSpikes, "creates spetial torus with some spikes" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_components_test_torus, MR::makeTorusWithComponents, "creates spetial torus without some segments" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_selfintersect_test_torus, MR::makeTorusWithSelfIntersections, "creates spetial torus with self-intersections" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, find_self_colliding_faces, MR::findSelfCollidingTrianglesBS, "fins FaceBitSet of self-intersections on mesh")

