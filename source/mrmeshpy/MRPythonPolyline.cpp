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



MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshSignedDistanceResult, []( pybind11::module_& m )
{

    pybind11::class_<MR::Polyline2>( m, "Polyline2" ).
        def_readwrite( "topology", &MR::Polyline2::topology ).
        def_readwrite( "points", &MR::Polyline2::points ).

        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Contours2f&>(), "Creates polyline from 2D contours, 3D polyline will get zero z - component" ).
        def( pybind11::init<const MR::Contours3f&>(), "Creates polyline from 3D contours, 2D polyline will lose z - component" ).

//         def( "addFromPoints", ( MR::EdgeId( MR::Polyline2::* )( const MR::Vector2f*, size_t, closed ) ) &MR::Polyline2::addFromPoints,
//             pybind11::arg( "vs" ), pybind11::arg( "num" ), pybind11::arg( "closed" ),
//             "Adds connected line in this, passing progressively via points *[vs, vs+num).\n"
//             "If closed argument is true then the last and the first points will be additionally connected.\n"
//             "Return the edge from first new to second new vertex." ).
//         def( "addFromPoints", ( MR::EdgeId( * )( const MR::Vector2f*, size_t ) ) &MR::Polyline2::addFromPoints,
//             pybind11::arg( "vs" ), pybind11::arg( "num" ),
//             "Adds connected line in this, passing progressively via points *[vs, vs+num).\n"
//             "If vs[0] == vs[num-1] then a closed line is created.\n"
//             "Return the edge from first new to second new vertex." ).
        def( "addPartByMask", &MR::Polyline2::addPartByMask,
            pybind11::arg( "from" ), pybind11::arg( "mask" ), pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr,
            "Appends polyline (from) in addition to this polyline: creates new edges, faces, verts and points." ).

        def( "orgPnt", &MR::Polyline2::orgPnt, pybind11::arg( "e" ), "Returns coordinates of the edge origin." ).
        def( "destPnt", &MR::Polyline2::destPnt, pybind11::arg( "e" ), "Returns coordinates of the edge destination." ).
        def( "edgePoint", &MR::Polyline2::edgePoint, pybind11::arg( "e" ), pybind11::arg( "f" ), "Returns a point on the edge: origin point for f=0 and destination point for f=1." ).
        def( "edgeCenter", &MR::Polyline2::edgeCenter, pybind11::arg( "e" ), "Returns edge's centroid." ).

        def( "edgeVector", &MR::Polyline2::edgeVector, pybind11::arg( "e" ), "Returns vector equal to edge destination point minus edge origin point." ).
        def( "edgeLength", &MR::Polyline2::edgeLength, pybind11::arg( "e" ), "Returns Euclidean length of the edge." ).
        def( "edgeLengthSq", &MR::Polyline2::edgeLengthSq, pybind11::arg( "e" ), "Returns squared Euclidean length of the edge (faster to compute than length)." ).
        def( "totalLength", &MR::Polyline2::totalLength, "Returns total length of the polyline." ).

        def( "getAABBTree", &MR::Polyline2::getAABBTree, "Returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner." ).
        def( "getAABBTreeNotCreate", &MR::Polyline2::getAABBTreeNotCreate, "Returns cached aabb-tree for this polyline, but does not create it if it did not exist." ).
        def( "getBoundingBox", &MR::Polyline2::getBoundingBox, "Returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())." ).
        def( "computeBoundingBox", &MR::Polyline2::computeBoundingBox, pybind11::arg( "toWorld" ) = nullptr,
            "Passes through all valid points and finds the minimal bounding box containing all of them.\n"
            "If toWorld transformation is given then returns minimal bounding box in world space." ).
        def( "transform", &MR::Polyline2::transform, pybind11::arg( "xf" ),
            "Returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())." ).

        def( "splitEdge", ( MR::EdgeId( MR::Polyline2::* )( MR::EdgeId, const MR::Vector2f& ) ) &MR::Polyline2::splitEdge,
            pybind11::arg( "EdgeId" ), pybind11::arg( "newVertPos" ),
            "Split given edge on two parts:\n"
            "dest(returned-edge) = org(e) - newly created vertex,\n"
            "org(returned-edge) = org(e-before-split),\n"
            "dest(e) = dest(e-before-split)" ).
        def( "splitEdge", ( MR::EdgeId( MR::Polyline2::* )( MR::EdgeId ) ) &MR::Polyline2::splitEdge, pybind11::arg( "EdgeId" ), "Split given edge on two equal parts" ).

        def( "invalidateCaches", &MR::Polyline2::invalidateCaches, "Invalidates caches (e.g. aabb-tree) after a change in polyline" ).

        def( "contours", &MR::Polyline2::contours, "Convert Polyline to simple contour structures with vector of points inside.\n"
            "If all even edges are consistently oriented, then the output contours will be oriented the same." ).
        def( "contours2", &MR::Polyline2::contours2, "Convert Polyline to simple 2D contour structures with vector of points inside.\n"
            "If all even edges are consistently oriented, then the output contours will be oriented the same." ).
        def( "toPolyline", ( MR::Polyline2( MR::Polyline2::* )() const ) &MR::Polyline2::toPolyline, "Convert Polyline2 to Polyline2 or vice versa" ).
        def( "toPolyline", ( MR::Polyline3( MR::Polyline2::* )( ) const )& MR::Polyline2::toPolyline, "Convert Polyline2 to Polyline2 or vice versa" ).

        def( "addFromEdgePath", &MR::Polyline2::addFromEdgePath, pybind11::arg( "mesh" ), pybind11::arg( "path" ),
            "Adds path to this polyline.\n"
            "Return the edge from first new to second new vertex." ).
        def( "addFromSurfacePath", &MR::Polyline2::addFromSurfacePath, pybind11::arg( "mesh" ), pybind11::arg( "path" ),
            "Adds path to this polyline.\n"
            "Return the edge from first new to second new vertex." );
} )


MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PlanarTriangulation, [] ( pybind11::module_& m )
{
    m.def( "triangulateContours", ( MR::Mesh( * )( const MR::Contours2f&, bool ) )& MR::PlanarTriangulation::triangulateContours,
        pybind11::arg( "contours" ), pybind11::arg( "mergeClosePoints" ) = true,
        "Triangulate 2d contours.\n"
        "Only closed contours are allowed (first point of each contour should be the same as last point of the contour).\n"
        "mergeClosePoints - merge close points in contours\n"
        "Return created mesh" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SymbolsMesh, [] ( pybind11::module_& m )
{
    m.def( "addBaseToPlanarMesh", &MR::addBaseToPlanarMesh,
        pybind11::arg( "mesh" ), pybind11::arg( "zOffset" ) = 1.0f,
        "Given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset to make a volumetric closed mesh.\n"
        "Note! zOffset should be > 0.\n" );
} )