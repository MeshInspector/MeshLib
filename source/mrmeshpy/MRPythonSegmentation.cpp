#include "MRPython/MRPython.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRContoursCut.h"
#include "MRMesh/MRSurroundingContour.h"
#include "MRMesh/MRFillContourByGraphCut.h"
#include "MRMesh/MRMeshTrimWithPlane.h"
#include "MRMesh/MREdgeMetric.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MRFillContour.h"
#include <pybind11/functional.h>

namespace MR
{

static void myTrimWithPlane( Mesh& mesh, const Plane3f & plane, MR::FaceMap* mapNew2Old )
{
    FaceHashMap new2OldHashMap;
    trimWithPlane( mesh, { .plane = plane }, { .new2Old = mapNew2Old ? &new2OldHashMap : nullptr } );
    if ( mapNew2Old )
    {
        for ( auto & [newF, oldF] : new2OldHashMap )
            mapNew2Old->autoResizeAt( newF ) = oldF;
    }
}

} //namespace MR

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Segmentation, [] ( pybind11::module_& m )
{
    m.def( "surroundingContour", MR::decorateExpected( []( const MR::Mesh & mesh, std::vector<MR::EdgeId> includeEdges, const MR::EdgeMetric & edgeMetric, const MR::Vector3f & dir )
        { return surroundingContour( mesh, std::move( includeEdges ), edgeMetric, dir ); } ),
        pybind11::arg( "mesh" ), pybind11::arg( "includeEdges" ), pybind11::arg( "edgeMetric" ), pybind11::arg( "dir" ),
        "Find the best closed edge loop passing through given edges, which minimizes the sum of given edge metric\n"
        "\tincludeEdges - contain all edges that must be present in the returned loop, probably with reversed direction (should have at least 2 elements)\n"
        "\tedgeMetric - returned loop will minimize this metric\n"
        "\tdir - direction approximately orthogonal to the loop" );

    m.def( "surroundingContour", MR::decorateExpected( []( const MR::Mesh & mesh, std::vector<MR::VertId> keyVertices, const MR::EdgeMetric & edgeMetric, const MR::Vector3f & dir )
        { return surroundingContour( mesh, std::move( keyVertices ), edgeMetric, dir ); } ),
        pybind11::arg( "mesh" ), pybind11::arg( "keyVertices" ), pybind11::arg( "edgeMetric" ), pybind11::arg( "dir" ),
        "Find the best closed edge loop passing through given vertices, which minimizes the sum of given edge metric\n"
        "\tkeyVertices - contain all vertices that returned loop must pass (should have at least 2 elements)\n"
        "\tedgeMetric - returned loop will minimize this metric\n"
        "\tdir - direction approximately orthogonal to the loop" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::EdgePath&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contour" ), pybind11::arg( "metric" ),
        "Fills region located to the left from given contour, by minimizing the sum of metric over the boundary" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const std::vector<MR::EdgePath>&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contours" ), pybind11::arg( "metric" ),
        "Fills region located to the left from given contours, by minimizing the sum of metric over the boundary" );

    m.def( "segmentByGraphCut", &MR::segmentByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "source" ), pybind11::arg( "sink" ), pybind11::arg( "metric" ),
        "Finds segment that divide mesh on source and sink (source included, sink excluded), by minimizing the sum of metric over the boundary" );

    m.def("cutMeshWithPlane", &MR::myTrimWithPlane,
        pybind11::arg( "mesh" ), pybind11::arg( "plane" ), pybind11::arg( "mapNew2Old" ) = nullptr,
        "This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal\n"
        "\tmesh - Input mesh that will be cut\n"
        "\tplane - Input plane to cut mesh with\n"
        "\tmapNew2Old - (this is optional output) map from newly generated faces to old faces (N-1)\n"
        "note: This function changes input mesh\n"
        "return: New edges that correspond to given contours" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, EdgeMetrics, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::EdgeMetric>( m, "EdgeMetric" );

    m.def( "identityMetric", &MR::identityMetric, "metric returning 1 for every edge" );
    m.def( "edgeLengthMetric", &MR::edgeLengthMetric, pybind11::arg( "mesh" ), "returns edge's length as a metric" );
    m.def( "edgeCurvMetric", &MR::edgeCurvMetric,
        pybind11::arg( "mesh" ), pybind11::arg( "angleSinFactor" ) = 2.0f, pybind11::arg( "angleSinForBoundary" ) = 0.0f,
        "returns edge's metric that depends both on edge's length and on the angle between its left and right faces\n"
        "\tangleSinFactor - multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)\n"
        "\tangleSinForBoundary - consider this dihedral angle sine for boundary edges" );
    m.def( "edgeTableSymMetric", &MR::edgeTableSymMetric, pybind11::arg( "topology" ), pybind11::arg( "metric" ), "pre-computes the metric for all mesh edges to quickly return it later for any edge; input metric must be symmetric: metric(e) == metric(e.sym())" );

    m.def( "buildShortestPath", ( MR::EdgePath( * )( const MR::Mesh&, MR::VertId, MR::VertId, float ) ) & MR::buildShortestPath,
        pybind11::arg( "mesh" ), pybind11::arg( "start" ), pybind11::arg( "finish" ), pybind11::arg( "maxPathLen" ) = FLT_MAX,
        "builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned" );
    m.def( "buildShortestPathBiDir", ( MR::EdgePath( * )( const MR::Mesh&, MR::VertId, MR::VertId, float ) ) & MR::buildShortestPathBiDir,
        pybind11::arg( "mesh" ), pybind11::arg( "start" ), pybind11::arg( "finish" ), pybind11::arg( "maxPathLen" ) = FLT_MAX,
        "builds shortest path in euclidean metric from start to finish vertices using faster search from both directions; if no path can be found then empty path is returned" );
    m.def( "buildSmallestMetricPath", ( MR::EdgePath( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::VertId, MR::VertId, float ) ) & MR::buildSmallestMetricPath,
        pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "start" ), pybind11::arg( "finish" ), pybind11::arg( "maxPathMetric" ) = FLT_MAX,
        "builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned" );
    m.def( "buildSmallestMetricPathBiDir", ( MR::EdgePath( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::VertId, MR::VertId, float ) ) & MR::buildSmallestMetricPathBiDir,
        pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "start" ), pybind11::arg( "finish" ), pybind11::arg( "maxPathMetric" ) = FLT_MAX,
        "builds shortest path in given metric from start to finish vertices using faster search from both directions; if no path can be found then empty path is returned" );
    m.def( "buildSmallestMetricPath", ( MR::EdgePath( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::VertId, const MR::VertBitSet&, float ) ) & MR::buildSmallestMetricPath,
        pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "start" ), pybind11::arg( "finish" ), pybind11::arg( "maxPathMetric" ) = FLT_MAX,
        "builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned" );

    m.def( "expand", ( void( * )( const MR::MeshTopology&, MR::VertBitSet&, int ) ) & MR::expand,
        pybind11::arg( "topology" ), pybind11::arg( "region" ), pybind11::arg( "hops" ) = 1,
        "adds to the region all vertices within given number of hops (stars) from the initial region boundary" );
    m.def( "expand", ( void( * )( const MR::MeshTopology&, MR::FaceBitSet&, int ) ) & MR::expand,
        pybind11::arg( "topology" ), pybind11::arg( "region" ), pybind11::arg( "hops" ) = 1,
        "adds to the region all faces within given number of hops (stars) from the initial region boundary" );
    m.def( "shrink", ( void( * )( const MR::MeshTopology&, MR::VertBitSet&, int ) ) & MR::shrink,
        pybind11::arg( "topology" ), pybind11::arg( "region" ), pybind11::arg( "hops" ) = 1,
        "removes from the region all vertices within given number of hops (stars) from the initial region boundary" );
    m.def( "shrink", ( void( * )( const MR::MeshTopology&, MR::FaceBitSet&, int ) ) & MR::shrink,
        pybind11::arg( "topology" ), pybind11::arg( "region" ), pybind11::arg( "hops" ) = 1,
        "removes from the region all faces within given number of hops (stars) from the initial region boundary" );

    m.def( "dilateRegionByMetric", ( bool( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::FaceBitSet&, float, MR::ProgressCallback ) ) & MR::dilateRegionByMetric,
       pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
       "expands the region (of faces or vertices) on given metric value" );
    m.def( "dilateRegionByMetric", ( bool( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::VertBitSet&, float, MR::ProgressCallback ) ) & MR::dilateRegionByMetric,
       pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
       "expands the region (of faces or vertices) on given metric value" );

    m.def( "erodeRegionByMetric", ( bool( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::FaceBitSet&, float, MR::ProgressCallback ) ) & MR::erodeRegionByMetric,
        pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "shrinks the region (of faces or vertices) on given metric value" );
    m.def( "erodeRegionByMetric", ( bool( * )( const MR::MeshTopology&, const MR::EdgeMetric&, MR::VertBitSet&, float, MR::ProgressCallback ) ) & MR::erodeRegionByMetric,
        pybind11::arg( "topology" ), pybind11::arg( "metric" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "shrinks the region (of faces or vertices) on given metric value" );

    m.def( "dilateRegion", ( bool( * )( const MR::Mesh&, MR::FaceBitSet&, float, MR::ProgressCallback ) ) & MR::dilateRegion,
        pybind11::arg( "mesh" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "expands the region (of faces or vertices) on given value (in meters)" );
    m.def( "dilateRegion", ( bool( * )( const MR::Mesh&, MR::VertBitSet&, float, MR::ProgressCallback ) ) & MR::dilateRegion,
        pybind11::arg( "mesh" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "expands the region (of faces or vertices) on given value (in meters)" );

    m.def( "erodeRegion", ( bool( * )( const MR::Mesh&, MR::FaceBitSet&, float, MR::ProgressCallback ) ) & MR::erodeRegion,
        pybind11::arg( "mesh" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "shrinks the region (of faces or vertices) on given value (in meters)" );
    m.def( "erodeRegion", ( bool( * )( const MR::Mesh&, MR::VertBitSet&, float, MR::ProgressCallback ) ) & MR::erodeRegion,
        pybind11::arg( "mesh" ), pybind11::arg( "region" ), pybind11::arg( "dilation" ), pybind11::arg( "callback" ) = MR::ProgressCallback{},
        "shrinks the region (of faces or vertices) on given value (in meters)" );

    m.def( "fillContourLeft", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::EdgePath& ) ) & MR::fillContourLeft,
        pybind11::arg( "topology" ), pybind11::arg( "contour" ), "fill region located to the left from given edges" );
    m.def( "fillContourLeft", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const std::vector<MR::EdgePath>& ) ) & MR::fillContourLeft,
        pybind11::arg( "topology" ), pybind11::arg( "contours" ), "fill region located to the left from given edges" );
} )
