#include "MRMesh/MRPython.h"
#include "MRMesh/MRPointCloud.h"
#include "MREAlgorithms/MREPointCloudTriangulation.h"
#include <pybind11/functional.h>

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, TriangulationParameters, []( pybind11::module_& m )
{
    pybind11::class_<MRE::TriangulationParameters>( m, "TriangulationParameters" ).
        def( pybind11::init<>() ).
        def_readwrite( "avgNumNeighbours", &MRE::TriangulationParameters::avgNumNeighbours ).
        def_readwrite( "critAngle", &MRE::TriangulationParameters::critAngle ).
        def_readwrite( "critHoleLength", &MRE::TriangulationParameters::critHoleLength );

    m.def( "triangulate_point_cloud", MRE::triangulatePointCloud,
        pybind11::arg( "pointCloud" ), pybind11::arg( "params" ), pybind11::arg( "progressCallback" ) = nullptr,
        "Creates mesh from given point cloud" );
} )
