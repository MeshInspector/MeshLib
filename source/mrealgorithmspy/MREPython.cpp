#include "MRMesh/MRPython.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRContoursCut.h"
#include "MRMesh/MRSurroundingContour.h"
#include "MRMesh/MRFillContourByGraphCut.h"

MR_INIT_PYTHON_MODULE_PRECALL( mrealgorithmspy, [] ()
{
    pybind11::module_::import( "mrmeshpy" );
} )

MR_ADD_PYTHON_FUNCTION( mrealgorithmspy, cut_mesh_with_plane, MR::cutMeshWithPlane, "cuts mesh with given plane" )

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, Segmentation, [] ( pybind11::module_& m )
{
    m.def( "surroundingContour", &MR::surroundingContour,
        pybind11::arg( "mesh" ), pybind11::arg( "includeEdgeOrgs" ), pybind11::arg( "edgeMetric" ), pybind11::arg( "dir" ),
        "Creating contour passing through given edges in given mesh" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const MR::EdgePath&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contour" ), pybind11::arg( "metric" ),
        "Fill region located to the left from given contour, by minimizing the sum of metric over the boundary" );

    m.def( "fillContourLeftByGraphCut", ( MR::FaceBitSet( * )( const MR::MeshTopology&, const std::vector<MR::EdgePath>&, const MR::EdgeMetric& ) )& MR::fillContourLeftByGraphCut,
        pybind11::arg( "topology" ), pybind11::arg( "contours" ), pybind11::arg( "metric" ),
        "Fill region located to the left from given contours, by minimizing the sum of metric over the boundary" );
} )