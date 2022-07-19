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

MR_ADD_PYTHON_FUNCTION( mrmeshpy, get_selected_mesh, pythonGetSelectedMesh, "gets mesh from selected ObjectMesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, set_mesh_to_selected, pythonSetMeshToSelected, "sets mesh to selected ObjectMesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, expand_verts, ( void( * )( const MeshTopology&, VertBitSet&, int ) )& expand, "expand vert bit set" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, shrink_verts, ( void( * )( const MeshTopology&, VertBitSet&, int ) )& shrink, "shrink vert bit set" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, expand_faces, ( void( * )( const MeshTopology&, FaceBitSet&, int ) )& expand, "expand face bit set" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, shrink_faces, ( void( * )( const MeshTopology&, FaceBitSet&, int ) )& shrink, "shrink face bit set" )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshTopology, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshTopology>( m, "MeshTopology" ).
        def( pybind11::init<>() ).
        def( "getValidFaces", &MR::MeshTopology::getValidFaces, pybind11::return_value_policy::copy ).
        def( "getValidVerts", &MR::MeshTopology::getValidVerts, pybind11::return_value_policy::copy ).
        def( "org", &MR::MeshTopology::org ).
        def( "dest", &MR::MeshTopology::dest ).
        def( "findBoundaryFaces", &MR::MeshTopology::findBoundaryFaces ).
        def( "findBoundaryEdges", &MR::MeshTopology::findBoundaryEdges ).
        def( "findBoundaryVerts", &MR::MeshTopology::findBoundaryVerts ).
        def( "findBoundary", &MR::MeshTopology::findBoundary ).
        def( "findHoleRepresentiveEdges", &MR::MeshTopology::findHoleRepresentiveEdges ).
        def( "getTriVerts", ( void( MR::MeshTopology::* )( FaceId, VertId&, VertId&, VertId& )const )& MR::MeshTopology::getTriVerts );
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

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilder, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshBuilder::Triangle>( m, "MeshBuilderTri").
        def( pybind11::init<VertId, VertId, VertId, FaceId>() );
    m.def( "topologyFromTriangles", &MR::MeshBuilder::fromTriangles, "constructs topology from given vecMeshBuilderTri" );
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
        def( "computeBoundingBox", ( Box3f( MR::Mesh::* )( const FaceBitSet*, const AffineXf3f* ) const )& MR::Mesh::computeBoundingBox ).
        def( "getBoundingBox", &MR::Mesh::getBoundingBox ).
        def( "area", ( double( MR::Mesh::* )( const FaceBitSet* fs )const )& MR::Mesh::area, pybind11::arg( "fs" ) = nullptr ).
        def( "volume", &MR::Mesh::volume, pybind11::arg( "region" ) = nullptr ).
        def( "pack", &MR::Mesh::pack, pybind11::arg( "outFmap" ) = nullptr, pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr, pybind11::arg( "rearrangeTriangles" ) = false ).
        def( "discreteMeanCurvature", &MR::Mesh::discreteMeanCurvature ).
        def_readwrite( "topology", &MR::Mesh::topology ).
        def_readwrite( "points", &MR::Mesh::points ).
        def( "triPoint", ( MR::Vector3f( MR::Mesh::* )( const MR::MeshTriPoint& )const )& MR::Mesh::triPoint ).
        def( "edgePoint", ( MR::Vector3f( MR::Mesh::* )( const MR::MeshEdgePoint& )const )& MR::Mesh::edgePoint ).
        def( "invalidateCaches", &MR::Mesh::invalidateCaches ).
        def( "transform", ( void( MR::Mesh::* ) ( const AffineXf3f& ) ) &MR::Mesh::transform );

    m.def( "copyMesh", &pythonCopyMeshFunction );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPart, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshPart>( m, "MeshPart" ).
        def( pybind11::init<const Mesh&>() ).
        def( pybind11::init<const Mesh&, const FaceBitSet*>() ).
        def_readwrite( "region", &MR::MeshPart::region );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVertBitSet, MR::VertBitSet )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceBitSet, MR::FaceBitSet )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, RegionBoundary, [] ( pybind11::module_& m )
{
    m.def( "getIncidentVerts", ( MR::VertBitSet( * )( const MR::MeshTopology&, const MR::FaceBitSet& ) )& MR::getIncidentVerts );
    m.def( "getInnerVerts", ( MR::VertBitSet( * )( const MR::MeshTopology&, const MR::FaceBitSet& ) )& MR::getInnerVerts );

    m.def( "getIncidentFaces", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::VertBitSet& ) )& MR::getIncidentFaces );
    m.def( "getInnerFaces", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::VertBitSet& ) )& MR::getInnerFaces );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshComponents, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::MeshComponents::FaceIncidence>( m, "FaceIncidence" ).
        value( "PerEdge", MR::MeshComponents::FaceIncidence::PerEdge ).
        value( "PerVertex", MR::MeshComponents::FaceIncidence::PerVertex );

    m.def( "get_mesh_components_verts", &MR::MeshComponents::getAllComponentsVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "get all vertices componensts of the mesh" );

    m.def( "get_mesh_components_face", &MR::MeshComponents::getAllComponents,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "get all faces componensts of the mesh" );

    m.def( "get_faces_component", &MR::MeshComponents::getComponent,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "faceId" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "get component of given face id" );

    m.def( "get_verts_component", &MR::MeshComponents::getComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "vertId" ),
        pybind11::arg( "region" ) = nullptr,
        "get component of given vert id" );

    m.def( "get_largest_faces_component", &MR::MeshComponents::getLargestComponent,
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MR::MeshComponents::FaceIncidence::PerEdge,
        "get largest faces component" );

    m.def( "get_largest_verts_component", &MR::MeshComponents::getLargestComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "get largest vertices component" );
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
    params.metric = getCircumscribedFillMetric( mesh );
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

