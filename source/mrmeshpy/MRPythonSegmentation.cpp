#include "MRMesh/MRPython.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRContoursCut.h"
#include "MRMesh/MRSurroundingContour.h"
#include "MRMesh/MRFillContourByGraphCut.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Segmentation, [] ( pybind11::module_& m )
{
    m.def( "surroundingContour", &MR::surroundingContour,
        pybind11::arg( "mesh" ), pybind11::arg( "includeEdges" ), pybind11::arg( "edgeMetric" ), pybind11::arg( "dir" ),
        "Find the best closed edge loop passing through given edges, \"best\" is according to given edge metric\n"
        "\tincludeEdges - contain all edges that must be present in the returned loop, probably with reversed direction (should have 2 or 3 elements)\n"
        "\tedgeMetric - returned loop will minimize this metric\n"
        "\tdir - direction approximately orthogonal to the loop" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::EdgePath&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contour" ), pybind11::arg( "metric" ),
        "Fills region located to the left from given contour, by minimizing the sum of metric over the boundary" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const std::vector<MR::EdgePath>&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contours" ), pybind11::arg( "metric" ),
        "Fills region located to the left from given contours, by minimizing the sum of metric over the boundary" );

    m.def( "segmentByGraphCut", MR::segmentByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "source" ), pybind11::arg( "sink" ), pybind11::arg( "metric" ),
        "Finds segment that divide mesh on source and sink (source included, sink excluded), by minimizing the sum of metric over the boundary" );

    m.def("cutMeshWithPlane",&MR::cutMeshWithPlane ,
        pybind11::arg( "mesh" ), pybind11::arg( "plane" ), pybind11::arg( "mapNew2Old" ) = nullptr,
        "This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal\n"
        "\tmesh - Input mesh that will be cut\n"
        "\tplane - Input plane to cut mesh with\n"
        "\tmapNew2Old - (this is optional output) map from newly generated faces to old faces (N-1)\n"
        "note: This function changes input mesh\n"
        "return: New edges that correspond to given contours" );
} )