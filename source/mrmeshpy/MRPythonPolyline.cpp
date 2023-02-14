#include "MRMesh/MRPython.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf2.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MR2DContoursTriangulation.h"
#include "MRMesh/MRSymbolMesh.h"
#include "MRMesh/MRFaceFace.h"
#include "MRMesh/MRPolyline2Collide.h"

#define CONCAT(a, b) 

#define MR_ADD_PYTHON_POLYLINE(dimension) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Polyline##dimension, []( pybind11::module_& m )\
{\
    using VectorType = MR::Vector##dimension<float>;\
    using PolylineType = MR::Polyline<VectorType>;\
    pybind11::class_<PolylineType>( m, "Polyline"#dimension ).\
/*        def_readwrite( "topology", &PolylineType::topology ).*/\
        def_readwrite( "points", &PolylineType::points ).\
\
        def( pybind11::init<>() ).\
        def( pybind11::init<const MR::Contours2f&>(), "Creates polyline from 2D contours, 3D polyline will get zero z - component" ).\
        def( pybind11::init<const MR::Contours3f&>(), "Creates polyline from 3D contours, 2D polyline will lose z - component" ).\
\
        def( "addFromPoints", ( MR::EdgeId( PolylineType::* )( const VectorType*, size_t, bool ) ) &PolylineType::addFromPoints,\
            pybind11::arg( "vs" ), pybind11::arg( "num" ), pybind11::arg( "closed" ),\
            "Adds connected line in this, passing progressively via points *[vs, vs+num).\n"\
            "If closed argument is true then the last and the first points will be additionally connected.\n"\
            "Return the edge from first new to second new vertex." ).\
        def( "addFromPoints", ( MR::EdgeId( PolylineType::* )( const VectorType*, size_t ) ) &PolylineType::addFromPoints,\
            pybind11::arg( "vs" ), pybind11::arg( "num" ),\
            "Adds connected line in this, passing progressively via points *[vs, vs+num).\n"\
            "If vs[0] == vs[num-1] then a closed line is created.\n"\
            "Return the edge from first new to second new vertex." ).\
        def( "addPartByMask", &PolylineType::addPartByMask,\
            pybind11::arg( "from" ), pybind11::arg( "mask" ), pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr,\
            "Appends polyline (from) in addition to this polyline: creates new edges, faces, verts and points." ).\
\
        def( "orgPnt", &PolylineType::orgPnt, pybind11::arg( "e" ), "Returns coordinates of the edge origin." ).\
        def( "destPnt", &PolylineType::destPnt, pybind11::arg( "e" ), "Returns coordinates of the edge destination." ).\
        def( "edgePoint", &PolylineType::edgePoint, pybind11::arg( "e" ), pybind11::arg( "f" ), "Returns a point on the edge: origin point for f=0 and destination point for f=1." ).\
        def( "edgeCenter", &PolylineType::edgeCenter, pybind11::arg( "e" ), "Returns edge's centroid." ).\
\
        def( "edgeVector", &PolylineType::edgeVector, pybind11::arg( "e" ), "Returns vector equal to edge destination point minus edge origin point." ).\
        def( "edgeSegment", &PolylineType::edgeSegment, pybind11::arg( "e" ), "Returns line segment of given edge." ).\
        def( "edgeLength", &PolylineType::edgeLength, pybind11::arg( "e" ), "Returns Euclidean length of the edge." ).\
        def( "edgeLengthSq", &PolylineType::edgeLengthSq, pybind11::arg( "e" ), "Returns squared Euclidean length of the edge (faster to compute than length)." ).\
        def( "totalLength", &PolylineType::totalLength, "Returns total length of the polyline." ).\
\
        def( "getAABBTree", &PolylineType::getAABBTree, "Returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner." ).\
        def( "getAABBTreeNotCreate", &PolylineType::getAABBTreeNotCreate, "Returns cached aabb-tree for this polyline, but does not create it if it did not exist." ).\
        def( "getBoundingBox", &PolylineType::getBoundingBox, "Returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())." ).\
        def( "computeBoundingBox", &PolylineType::computeBoundingBox, pybind11::arg( "toWorld" ) = nullptr,\
            "Passes through all valid points and finds the minimal bounding box containing all of them.\n"\
            "If toWorld transformation is given then returns minimal bounding box in world space." ).\
        def( "transform", &PolylineType::transform, pybind11::arg( "xf" ),\
            "Returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())." ).\
\
        def( "splitEdge", ( MR::EdgeId( PolylineType::* )( MR::EdgeId, const VectorType& ) ) &PolylineType::splitEdge,\
            pybind11::arg( "EdgeId" ), pybind11::arg( "newVertPos" ),\
            "Split given edge on two parts:\n"\
            "dest(returned-edge) = org(e) - newly created vertex,\n"\
            "org(returned-edge) = org(e-before-split),\n"\
            "dest(e) = dest(e-before-split)" ).\
        def( "splitEdge", ( MR::EdgeId( PolylineType::* )( MR::EdgeId ) ) &PolylineType::splitEdge, pybind11::arg( "EdgeId" ), "Split given edge on two equal parts" ).\
\
        def( "invalidateCaches", &PolylineType::invalidateCaches, "Invalidates caches (e.g. aabb-tree) after a change in polyline" ).\
\
        def( "contours", &PolylineType::contours, "Convert Polyline to simple contour structures with vector of points inside.\n"\
            "If all even edges are consistently oriented, then the output contours will be oriented the same." ).\
        def( "contours2", &PolylineType::contours2, "Convert Polyline to simple 2D contour structures with vector of points inside.\n"\
            "If all even edges are consistently oriented, then the output contours will be oriented the same." ).\
        def( "toPolyline", ( MR::Polyline2( PolylineType::* )() const ) &PolylineType::toPolyline, "Convert Polyline##dimension to Polyline2." ).\
        def( "toPolyline", ( MR::Polyline3( PolylineType::* )() const ) &PolylineType::toPolyline, "Convert Polyline##dimension to Polyline3." ).\
\
        def( "addFromEdgePath", &PolylineType::addFromEdgePath, pybind11::arg( "mesh" ), pybind11::arg( "path" ),\
            "Adds path to this polyline.\n"\
            "Return the edge from first new to second new vertex." ).\
        def( "addFromSurfacePath", &PolylineType::addFromSurfacePath, pybind11::arg( "mesh" ), pybind11::arg( "path" ),\
            "Adds path to this polyline.\n"\
            "Return the edge from first new to second new vertex." );\
} )

MR_ADD_PYTHON_POLYLINE(2)
MR_ADD_PYTHON_POLYLINE(3)

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PlanarTriangulation, [] ( pybind11::module_& m )
{
    m.def("findHoleVertIdsByHoleEdges",&MR::PlanarTriangulation::findHoleVertIdsByHoleEdges,
        pybind11::arg( "tp" ), pybind11::arg( "holePaths" ),
        "return vertices of holes that correspond internal contours representation of PlanarTriangulation" );

    m.def( "triangulateContours", ( MR::Mesh( * )( const MR::Contours2f&, const MR::PlanarTriangulation::HolesVertIds* ) )& MR::PlanarTriangulation::triangulateContours,
        pybind11::arg( "contours" ), pybind11::arg( "holeVertsIds" ) = nullptr,
        "Triangulate 2d contours.\n"
        "Only closed contours are allowed (first point of each contour should be the same as last point of the contour).\n"
        "holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates\n"
        "Return created mesh" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SymbolsMesh, [] ( pybind11::module_& m )
{
    m.def( "addBaseToPlanarMesh", &MR::addBaseToPlanarMesh,
        pybind11::arg( "mesh" ), pybind11::arg( "zOffset" ) = 1.0f,
        "Given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset to make a volumetric closed mesh.\n"
        "Note! zOffset should be > 0.\n" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, UndirectedEdgeUndirectedEdge, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::UndirectedEdgeUndirectedEdge>( m, "UndirectedEdgeUndirectedEdge" ).
        def( pybind11::init<>() ).
        def( pybind11::init<MR::UndirectedEdgeId, MR::UndirectedEdgeId>(), pybind11::arg( "a" ), pybind11::arg( "b" ) ).
        def_readwrite( "aUndirEdge", &MR::UndirectedEdgeUndirectedEdge::aUndirEdge ).
        def_readwrite( "bUndirEdge", &MR::UndirectedEdgeUndirectedEdge::bUndirEdge );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorUndirectedEdgeUndirectedEdge, MR::UndirectedEdgeUndirectedEdge )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, findSelfCollidingEdges, [] ( pybind11::module_& m )
{
    m.def( "findSelfCollidingEdges", &MR::findSelfCollidingEdges, pybind11::arg( "polyline" ), "finds all pairs of colliding edges from 2d polyline" );
} )
