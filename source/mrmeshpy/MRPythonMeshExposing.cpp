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
#include "MRMesh/MRSphere.h"
#include "MRMesh/MRUVSphere.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRExpected.h"
#include <pybind11/functional.h>
#include "MRMesh/MRPartMapping.h"
using namespace MR;

Mesh pythonGetSelectedMesh()
{
    auto selected = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return {};
    if ( !selected[0] || !selected[0]->mesh() )
        return {};
    return *selected[0]->mesh();
}

void pythonSetMeshToSelected( Mesh mesh )
{
    auto selected = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
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
    pybind11::class_<MeshTopology>( m, "MeshTopology" ).
        def( pybind11::init<>() ).
        def( "numValidFaces", &MeshTopology::numValidFaces, "returns the number of valid faces" ).
        def( "numValidVerts", &MeshTopology::numValidVerts, "returns the number of valid vertices" ).
        def( "getValidFaces", &MeshTopology::getValidFaces, pybind11::return_value_policy::copy, "returns cached set of all valid faces" ).
        def( "getValidVerts", &MeshTopology::getValidVerts, pybind11::return_value_policy::copy, "returns cached set of all valid vertices" ).
        def( "flip", (void (MeshTopology::*)(FaceBitSet&)const)&MeshTopology::flip, pybind11::arg( "fs" ), "sets in (fs) all valid faces that were not selected before the call, and resets other bits" ).
        def( "flip", (void (MeshTopology::*)(VertBitSet&)const)&MeshTopology::flip, pybind11::arg( "vs" ), "sets in (vs) all valid vertices that were not selected before the call, and resets other bits" ).
        def( "flipOrientation", &MeshTopology::flipOrientation, "flip orientation (normals) of all faces" ).
        def( "org", &MeshTopology::org, pybind11::arg( "he" ), "returns origin vertex of half-edge" ).
        def( "dest", &MeshTopology::dest, pybind11::arg( "he" ), "returns destination vertex of half-edge" ).
        def( "findBoundaryFaces", &MeshTopology::findBoundaryFaces, "returns all boundary faces, having at least one boundary edge" ).
        def( "findBoundaryEdges", &MeshTopology::findBoundaryEdges, "returns all boundary edges, where each edge does not have valid left face" ).
        def( "findBoundaryVerts", &MeshTopology::findBoundaryVerts, "returns all boundary vertices, incident to at least one boundary edge" ).
        def( "deleteFaces", &MeshTopology::deleteFaces, pybind11::arg( "fs" ), "deletes multiple given faces" ).
        def( "findHoleRepresentiveEdges", &MeshTopology::findHoleRepresentiveEdges, "returns one edge with no valid left face for every boundary in the mesh" ).
        def( "getTriVerts", ( void( MeshTopology::* )( FaceId, VertId&, VertId&, VertId& )const )& MeshTopology::getTriVerts,
            pybind11::arg("f"), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ), 
            "gets 3 vertices of given triangular face;\n"
            "the vertices are returned in counter-clockwise order if look from mesh outside" ).
        def( "edgeSize", &MeshTopology::edgeSize, "returns the number of half-edge records including lone ones" ).
        def( "undirectedEdgeSize", &MeshTopology::undirectedEdgeSize, "returns the number of undirected edges (pairs of half-edges) including lone ones" ).
        def( "computeNotLoneUndirectedEdges", &MeshTopology::computeNotLoneUndirectedEdges, "computes the number of not-lone (valid) undirected edges" ).
        def( pybind11::self == pybind11::self, "compare that two topologies are exactly the same" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Vector, [] ( pybind11::module_& m )
{
    pybind11::class_<VertCoords>( m, "VertCoords" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertCoords::vec_ );

    pybind11::class_<FaceMap>( m, "FaceMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &FaceMap::vec_ );

    pybind11::class_<VertMap>( m, "VertMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertMap::vec_ );

    pybind11::class_<WholeEdgeMap>( m, "WholeEdgeMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &WholeEdgeMap::vec_ );

    pybind11::class_<UndirectedEdgeMap>( m, "UndirectedEdgeMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &UndirectedEdgeMap::vec_ );

    pybind11::class_<EdgeMap>( m, "EdgeMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &EdgeMap::vec_ );

    pybind11::class_<VertScalars>( m, "VectorFloatByVert" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertScalars::vec_ );

    pybind11::class_<FaceNormals>( m, "FaceNormals" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &FaceNormals::vec_ );

    pybind11::class_<Vector<Vector2f, VertId>>( m, "VertCoords2" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &Vector<Vector2f, VertId>::vec_ );

    pybind11::class_<VertColors>( m, "VertColorMap" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertColors::vec_ );
} )

MR_ADD_PYTHON_MAP( mrmeshpy, FaceHashMap, FaceHashMap )
MR_ADD_PYTHON_MAP( mrmeshpy, VertHashMap, VertHashMap )
MR_ADD_PYTHON_MAP( mrmeshpy, WholeEdgeHashMap, WholeEdgeHashMap )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PartMapping, [] ( pybind11::module_& m )
{
    pybind11::class_<PartMapping>( m, "PartMapping", "mapping among elements of source mesh, from which a part is taken, and target (this) mesh" ).
        def( pybind11::init<>() ).
        def_readwrite( "src2tgtFaces", &PartMapping::src2tgtFaces, "from.id -> this.id" ).
        def_readwrite( "src2tgtVerts", &PartMapping::src2tgtVerts, "from.id -> this.id" ).
        def_readwrite( "src2tgtEdges", &PartMapping::src2tgtEdges, "from.id -> this.id" ).
        def_readwrite( "tgt2srcFaces", &PartMapping::tgt2srcFaces, "this.id -> from.id" ).
        def_readwrite( "tgt2srcVerts", &PartMapping::tgt2srcVerts, "this.id -> from.id" ).
        def_readwrite( "tgt2srcEdges", &PartMapping::tgt2srcEdges, "this.id -> from.id" );
} )


MeshTopology topologyFromTriangles( const Triangulation& t, const MeshBuilder::BuildSettings& s )
{
    return MeshBuilder::fromTriangles( t, s );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilder, []( pybind11::module_& m )
{
    pybind11::class_<ThreeVertIds>( m, "ThreeVertIds" ).
        def( pybind11::init( [] ( VertId v0, VertId v1, VertId v2 )->ThreeVertIds
    {
        return { v0, v1, v2 };
    } ) );

    pybind11::class_<Triangulation>( m, "Triangulation" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &Triangulation::vec_ );

    pybind11::class_<MeshBuilder::BuildSettings>( m, "MeshBuilderSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "region", &MeshBuilder::BuildSettings::region, "if region is given then on input it contains the faces to be added, and on output the faces failed to be added" ).
        def_readwrite( "shiftFaceId", &MeshBuilder::BuildSettings::shiftFaceId, "this value to be added to every faceId before its inclusion in the topology" ).
        def_readwrite( "allowNonManifoldEdge", &MeshBuilder::BuildSettings::allowNonManifoldEdge, "whether to permit non-manifold edges in the resulting topology" );

    m.def( "topologyFromTriangles",
        ( MeshTopology( * )( const Triangulation&, const MeshBuilder::BuildSettings& ) )& topologyFromTriangles,
        pybind11::arg( "triangulation" ), pybind11::arg( "settings" ) = MeshBuilder::BuildSettings{},
        "construct mesh topology from a set of triangles with given ids;\n"
        "if skippedTris is given then it receives all input triangles not added in the resulting topology" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorThreeVertIds, ThreeVertIds )

Mesh pythonCopyMeshFunction( const Mesh& mesh )
{
    return mesh;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Mesh, [] ( pybind11::module_& m )
{
    pybind11::class_<Mesh>( m, "Mesh" ).
        def( pybind11::init<>() ).
        def( "computeBoundingBox", ( Box3f( Mesh::* )( const FaceBitSet*, const AffineXf3f* ) const )& Mesh::computeBoundingBox,
            pybind11::arg( "region" ) = nullptr, pybind11::arg( "toWorld" ) = nullptr,
            "passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them\n"
            "if toWorld transformation is given then returns minimal bounding box in world space" ).
        def( "getBoundingBox", &Mesh::getBoundingBox,
            "returns the bounding box containing all valid vertices (implemented via getAABBTree())\n"
            "this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision" ).
        def( "area", ( double( Mesh::* )( const FaceBitSet* fs )const )& Mesh::area, pybind11::arg( "fs" ) = nullptr, "this version returns the area of whole mesh if argument is nullptr" ).
        def( "volume", &Mesh::volume, pybind11::arg( "region" ) = nullptr,
            "returns volume of closed mesh region, if region is not closed DBL_MAX is returned\n"
            "if region is nullptr - whole mesh is region" ).
        def( "pack", &Mesh::pack, pybind11::arg( "outFmap" ) = nullptr, pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr, pybind11::arg( "rearrangeTriangles" ) = false,
            "tightly packs all arrays eliminating lone edges and invalid face, verts and points,\n"
            "optionally returns mappings: old.id -> new.id" ).
        def( "discreteMeanCurvature", ( float( Mesh::* )( VertId ) const ) &Mesh::discreteMeanCurvature, pybind11::arg( "v" ),
            "computes discrete mean curvature in given vertex measures in length^-1;\n"
            "0 for planar regions, positive for convex surface, negative for concave surface" ).
        def_readwrite( "topology", &Mesh::topology ).
        def_readwrite( "points", &Mesh::points ).
        def( "triPoint", ( Vector3f( Mesh::* )( const MeshTriPoint& )const )& Mesh::triPoint, pybind11::arg( "p" ), "returns interpolated coordinates of given point" ).
        def( "edgePoint", ( Vector3f( Mesh::* )( const MeshEdgePoint& )const )& Mesh::edgePoint, pybind11::arg( "ep" ), "returns a point on the edge: origin point for f=0 and destination point for f=1" ).
        def( "invalidateCaches", &Mesh::invalidateCaches, "Invalidates caches (e.g. aabb-tree) after a change in mesh geometry or topology" ).
        def( "transform", ( void( Mesh::* ) ( const AffineXf3f&, const VertBitSet* ) )& Mesh::transform, pybind11::arg( "xf" ), pybind11::arg( "region" ) = nullptr,
             "applies given transformation to specified vertices\n"
             "if region is nullptr, all valid mesh vertices are used" ).

        def( "splitEdge", ( EdgeId( Mesh::* )( EdgeId, const Vector3f&, FaceBitSet*, FaceHashMap* ) )& Mesh::splitEdge, 
            pybind11::arg( "e" ), pybind11::arg( "newVertPos" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given edge on two parts:\n"
            "dest(returned-edge) = org(e) - newly created vertex,\n"
            "org(returned-edge) = org(e-before-split),\n"
            "dest(e) = dest(e-before-split)\n"
            "\tleft and right faces of given edge if valid are also subdivided on two parts each;\n"
            "\tif left or right faces of the original edge were in the region, then include new parts of these faces in the region\n"
            "\tnew2Old - receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitEdge", ( EdgeId( Mesh::* )( EdgeId, FaceBitSet*, FaceHashMap* ) )& Mesh::splitEdge,
            pybind11::arg( "e" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given edge on two equal parts:\n"
            "dest(returned-edge) = org(e) - newly created vertex,\n"
            "org(returned-edge) = org(e-before-split),\n"
            "dest(e) = dest(e-before-split)\n"
            "\tleft and right faces of given edge if valid are also subdivided on two parts each;\n"
            "\tif left or right faces of the original edge were in the region, then include new parts of these faces in the region\n"
            "\tnew2Old - receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitFace", ( VertId( Mesh::* )( FaceId, const Vector3f&, FaceBitSet*, FaceHashMap* ) )& Mesh::splitFace,
            pybind11::arg( "f" ), pybind11::arg( "newVertPos" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given triangle on three triangles, introducing new vertex with given coordinates and connecting it to original triangle vertices;\n"
            "if region is given, then it must include (f) and new faces will be added there as well\n"
            "\tnew2Old receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitFace", ( VertId( Mesh::* )( FaceId, FaceBitSet*, FaceHashMap* ) )& Mesh::splitFace,
            pybind11::arg( "f" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given triangle on three triangles, introducing new vertex in the centroid of original triangle and connecting it to original triangle vertices;\n"
            "if region is given, then it must include (f) and new faces will be added there as well\n"
            "\tnew2Old receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "addPartByMask", ( void( Mesh::* )( const Mesh&, const FaceBitSet&, const PartMapping& ) )& Mesh::addPartByMask,
            pybind11::arg( "from" ), pybind11::arg( "fromFaces" ) = nullptr, pybind11::arg( "map" ) = PartMapping{},
            "appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points\n"
            "copies only portion of (from) specified by fromFaces" ).

        def( pybind11::self == pybind11::self, "compare that two meshes are exactly the same" );

    m.def( "copyMesh", &pythonCopyMeshFunction, pybind11::arg( "mesh" ), "returns copy of input mesh" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPart, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshPart>( m, "MeshPart", "stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)" ).
        def( pybind11::init<const Mesh&, const FaceBitSet*>(), pybind11::arg( "mesh" ), pybind11::arg( "region" ) = nullptr ).
        def_readwrite( "region", &MeshPart::region, "nullptr here means whole mesh" );

    pybind11::implicitly_convertible<const Mesh&, MeshPart>();
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVertBitSet, VertBitSet )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceBitSet, FaceBitSet )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, RegionBoundary, [] ( pybind11::module_& m )
{
    m.def( "findLeftBoundary", ( std::vector<EdgeLoop>( * )( const MeshTopology&, const FaceBitSet* ) )& findLeftBoundary,
        pybind11::arg( "topology" ), pybind11::arg( "region" ) = nullptr,
        "returns all region boundary loops;\n"
        "every loop has region faces on the left, and not-region faces or holes on the right" );

    m.def( "findRightBoundary", ( std::vector<EdgeLoop>( * )( const MeshTopology&, const FaceBitSet* ) )& findRightBoundary,
        pybind11::arg( "topology" ), pybind11::arg( "region" ) = nullptr,
        "returns all region boundary loops;\n"
        "every loop has region faces on the right, and not-region faces or holes on the left" );

    m.def( "getIncidentVerts", ( VertBitSet( * )( const MeshTopology&, const FaceBitSet& ) )& getIncidentVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices incident to given faces" );
    m.def( "getInnerVerts", ( VertBitSet( * )( const MeshTopology&, const FaceBitSet& ) )& getInnerVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices with all their faces in given set" );

    m.def( "getIncidentFaces", ( FaceBitSet( * )( const MeshTopology&, const VertBitSet& ) )& getIncidentFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces incident to given vertices" );
    m.def( "getInnerFaces", ( FaceBitSet( * )( const MeshTopology&, const VertBitSet& ) )& getInnerFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces with all their vertices in given set" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshComponents, [] ( pybind11::module_& m )
{
    pybind11::enum_<MeshComponents::FaceIncidence>( m, "FaceIncidence", "Face incidence type" ).
        value( "PerEdge", MeshComponents::FaceIncidence::PerEdge, "face can have neighbor only via edge" ).
        value( "PerVertex", MeshComponents::FaceIncidence::PerVertex, "face can have neighbor via vertex" );

    m.def( "getAllComponentsVerts", &MeshComponents::getAllComponentsVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "gets all connected components of mesh part" );

    m.def( "getAllComponents", []( const MeshPart& meshPart, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getAllComponents( meshPart, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MeshComponents::FaceIncidence::PerEdge,
        "gets all connected components of mesh part" );

     m.def( "getComponent", []( const MeshPart& meshPart, FaceId id, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getComponent( meshPart, id, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg( "faceId" ),
        pybind11::arg( "incidence" ) = MeshComponents::FaceIncidence::PerEdge,
        "returns one connected component containing given face,\n"
        "not effective to call more than once, if several components are needed use getAllComponents" );

    m.def( "getComponentVerts", &MeshComponents::getComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "vertId" ),
        pybind11::arg( "region" ) = nullptr,
        "returns one connected component containing given vertex,\n"
        "not effective to call more than once, if several components are needed use getAllComponentsVerts" );

     m.def( "getLargestComponent", []( const MeshPart& meshPart, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getLargestComponent( meshPart, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg( "incidence" ) = MeshComponents::FaceIncidence::PerEdge,
        "returns largest by surface area component" );

    m.def( "getLargestComponentVerts", &MeshComponents::getLargestComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "returns largest by number of elements component" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorMesh, Mesh )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FillHole, [] ( pybind11::module_& m )
{
    pybind11::enum_<FillHoleParams::MultipleEdgesResolveMode>( m, "FillHoleParamsMultipleEdgesResolveMode" ).
        value( "None", FillHoleParams::MultipleEdgesResolveMode::None, "do not avoid multiple edges" ).
        value( "Simple", FillHoleParams::MultipleEdgesResolveMode::Simple, "avoid creating edges that already exist in topology (default)" ).
        value( "Strong", FillHoleParams::MultipleEdgesResolveMode::Strong,
            "makes additional efforts to avoid creating multiple edges,\n"
            "in some rare cases it is not possible (cases with extremely bad topology),\n"
            "if you faced one try to use duplicateMultiHoleVertices before fillHole" );

    pybind11::class_<FillHoleMetric>( m, "FillHoleMetric", "This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions" ).
        def( pybind11::init<>() );

    m.def( "getPlaneFillMetric", &getPlaneFillMetric, pybind11::arg( "mesh" ), pybind11::arg( "e" ),
        "Same as getCircumscribedFillMetric, but with extra penalty for the triangles having\n"
        "normals looking in the opposite side of plane containing left of (e)." );
    
    m.def( "getEdgeLengthFillMetric", &getEdgeLengthFillMetric, pybind11::arg( "mesh" ),
        "Simple metric minimizing the sum of all edge lengths" );
    
    m.def( "getEdgeLengthStitchMetric", &getEdgeLengthStitchMetric, pybind11::arg( "mesh" ),
        "Forbids connecting vertices from the same hole\n"
        "Simple metric minimizing edge length" );

    m.def( "getCircumscribedMetric", &getCircumscribedMetric, pybind11::arg( "mesh" ),
        "This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.\n"
        "It is rather fast to calculate, and it results in typically good triangulations." );

    m.def( "getUniversalMetric", &getUniversalMetric, pybind11::arg( "mesh" ),
        "This metric minimizes the maximal dihedral angle between the faces in the triangulation\n"
        "and on its boundary, and it avoids creating too degenerate triangles;\n"
        " for planar holes it is the same as getCircumscribedMetric" );

    m.def( "getComplexFillMetric", &getComplexFillMetric, pybind11::arg( "mesh" ), pybind11::arg( "e" ),
        "This metric minimizes the sum of triangleMetric for all triangles in the triangulation\n"
        "plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n"
        "Where\n"
        "triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n"
        "edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4." );

    m.def( "getMinAreaMetric", &getMinAreaMetric, pybind11::arg( "mesh" ),
        "This metric is for triangulation construction with minimal summed area of triangles.\n"
        "Warning: this metric can produce degenerated triangles" );

    pybind11::class_<FillHoleParams>( m, "FillHoleParams", "Structure has some options to control fillHole" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &FillHoleParams::metric, "Specifies triangulation metric\n""default for fillHole: getCircumscribedFillMetric" ).
        def_readwrite( "multipleEdgesResolveMode", &FillHoleParams::multipleEdgesResolveMode ).
        def_readwrite( "makeDegenerateBand", &FillHoleParams::makeDegenerateBand,
            "If true creates degenerate faces band around hole to have sharp angle visualization\n"
            "warning: This flag bad for result topology, most likely you do not need it" ).
        def_readwrite( "maxPolygonSubdivisions", &FillHoleParams::maxPolygonSubdivisions, "The maximum number of polygon subdivisions on a triangle and two smaller polygons,\n""must be 2 or larger" ).
        def_readwrite( "outNewFaces", &FillHoleParams::outNewFaces, "If not nullptr accumulate new faces" );

    pybind11::class_<StitchHolesParams>( m, "StitchHolesParams", "Structure has some options to control buildCylinderBetweenTwoHoles" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &StitchHolesParams::metric,
            "Specifies triangulation metric\n"
            "default for buildCylinderBetweenTwoHoles: getComplexStitchMetric").
        def_readwrite( "outNewFaces", &StitchHolesParams::outNewFaces, "If not nullptr accumulate new faces" );

    m.def( "fillHole", &fillHole,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "params" ) = FillHoleParams{},
        "Fills given hole represented by one of its edges (having no valid left face),\n"
        "uses fillHoleTrivially if cannot fill hole without multiple edges,\n"
        "default metric: CircumscribedFillMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents hole\n"
        "\tparams - parameters of hole filling" );

    m.def( "buildCylinderBetweenTwoHoles", ( void ( * )( Mesh&, EdgeId, EdgeId, const StitchHolesParams& ) )& buildCylinderBetweenTwoHoles,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "params" ) = StitchHolesParams{},
        "Build cylindrical patch to fill space between two holes represented by one of their edges each,\n"
        "default metric: ComplexStitchMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents 1st hole\n"
        "\tb - EdgeId which represents 2nd hole\n"
        "\tparams - parameters of holes stitching" );

    m.def( "buildCylinderBetweenTwoHoles", ( bool ( * )( Mesh&, const StitchHolesParams& ) )& buildCylinderBetweenTwoHoles,
       pybind11::arg( "mesh" ), pybind11::arg( "params" ) = StitchHolesParams{},
       "this version finds holes in the mesh by itself and returns false if they are not found" );
} )

Mesh pythonMergeMehses( const pybind11::list& meshes )
{
    Mesh res;
    for ( int i = 0; i < pybind11::len( meshes ); ++i )
        res.addPart( pybind11::cast<Mesh>( meshes[i] ) );
    return res;
}

FaceBitSet getFacesByMinEdgeLength( const Mesh& mesh, float minLength )
{
    using namespace MR;
    FaceBitSet resultFaces( mesh.topology.getValidFaces().size() );
    float minLengthSq = minLength * minLength;
    for ( auto ue : undirectedEdges( mesh.topology ) )
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
    m.def( "computePerVertNormals", &computePerVertNormals, pybind11::arg( "mesh" ), "returns a vector with vertex normals in every element for valid mesh vertices" );
    m.def( "computePerFaceNormals", &computePerFaceNormals, pybind11::arg( "mesh" ), "returns a vector with face-normal in every element for valid mesh faces" );
    m.def( "mergeMehses", &pythonMergeMehses, pybind11::arg( "meshes" ), "merge python list of meshes to one mesh" );
    m.def( "getFacesByMinEdgeLength", &getFacesByMinEdgeLength, pybind11::arg( "mesh" ), pybind11::arg( "minLength" ), "return faces with at least one edge longer than min edge length" );
    m.def( "buildBottom", &buildBottom, pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "dir" ), pybind11::arg( "holeExtension" ), pybind11::arg( "outNewFaces" ) = nullptr,
        "adds cylindrical extension of given hole represented by one of its edges (having no valid left face)\n"
        "by adding new vertices located in lowest point of the hole -dir*holeExtension and 2 * number_of_hole_edge triangles;\n"
        "return: the edge of new hole opposite to input edge (a)" );

    m.def( "makeCube", &makeCube, pybind11::arg( "size" ) = Vector3f::diagonal( 1 ), pybind11::arg( "base" ) = Vector3f::diagonal( -0.5f ),
        "Base is \"lower\" corner of the cube coordinates" );
    
    pybind11::class_<SphereParams>( m, "SphereParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "radius", &SphereParams::radius ).
        def_readwrite( "numMeshVertices", &SphereParams::numMeshVertices );

    m.def( "makeSphere", &makeSphere, pybind11::arg( "params" ),
        "creates a mesh of sphere with irregular triangulation" );
    m.def( "makeUVSphere", &makeUVSphere, 
        pybind11::arg( "radius" ) = 1.0f, 
        pybind11::arg( "horisontalResolution" ) = 16, pybind11::arg( "verticalResolution" ) = 16,
        "Z is polar axis of this UVSphere" );

    m.def( "makeCylinder", &makeCylinder,
        pybind11::arg( "radius" ) = 0.1f,
        pybind11::arg( "length" ) = 1.0f,
        pybind11::arg( "resolution" ) = 16,
        "creates Z-looking cylinder with radius 'radius', height - 'length', its base have 'resolution' sides" );

    m.def( "makeCylinderAdvanced", &makeCylinderAdvanced,
        pybind11::arg( "radius0" ) = 0.1f,
        pybind11::arg( "radius1" ) = 0.1f,
        pybind11::arg( "start_angle" ) = 0.0f,
        pybind11::arg( "arc_size" ) = 2.0f * PI_F,
        pybind11::arg( "length" ) = 1.0f,
        pybind11::arg( "resolution" ) = 16 );

    m.def( "makeTorus", &makeTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "Z is symmetry axis of this torus\n"
        "points - optional out points of main circle" );

    m.def( "makeOuterHalfTorus", &makeOuterHalfTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus without inner half faces\n"
        "main application - testing fillHole and Stitch" );

    m.def( "makeTorusWithUndercut", &makeTorusWithUndercut,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with inner protruding half as undercut\n"
        "main application - testing fixUndercuts" );

    m.def( "makeTorusWithSpikes", &makeTorusWithSpikes,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with some handed-up points\n"
        "main application - testing fixSpikes and Relax" );

    m.def( "makeTorusWithComponents", &makeTorusWithComponents,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );

    m.def( "makeTorusWithSelfIntersections", &makeTorusWithSelfIntersections,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );


    pybind11::class_<FaceFace>( m, "FaceFace" ).
        def( pybind11::init<>() ).
        def( pybind11::init<FaceId, FaceId>(), pybind11::arg( "a" ), pybind11::arg( "b" ) ).
        def_readwrite( "aFace", &FaceFace::aFace ).
        def_readwrite( "bFace", &FaceFace::bFace );

    m.def( "findSelfCollidingTriangles", decorateExpected( &findSelfCollidingTriangles ), pybind11::arg( "mp" ), pybind11::arg( "cb" ) = ProgressCallback{}, "finds all pairs of colliding triangles from one mesh or a region" );

    m.def( "findSelfCollidingTrianglesBS", decorateExpected( &findSelfCollidingTrianglesBS ), pybind11::arg( "mp" ), pybind11::arg( "cb" ) = ProgressCallback{}, "finds union of all self-intersecting faces" );

    m.def( "findCollidingTriangles", &findCollidingTriangles, 
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "firstIntersectionOnly" ) = false, 
        "finds all pairs of colliding triangles from two meshes or two mesh regions\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tfirstIntersectionOnly - if true then the function returns at most one pair of intersecting triangles and returns faster" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceFace, FaceFace )
