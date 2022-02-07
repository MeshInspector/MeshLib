#include "MRMesh/MREmbeddedPython.h"
#include "MRMesh/MRPointCloud.h"
#include "MREAlgorithms/MREPointCloudTriangulation.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, TriangulationParameters, []( pybind11::module_& m )
{
    pybind11::class_<MRE::TriangulationParameters>( m, "TriangulationParameters" ).
        def( pybind11::init<>() ).
        def_readwrite( "avgNumNeighbours", &MRE::TriangulationParameters::avgNumNeighbours ).
        def_readwrite( "critAngle", &MRE::TriangulationParameters::critAngle ).
        def_readwrite( "critHoleLength", &MRE::TriangulationParameters::critHoleLength );
} )

MR_ADD_PYTHON_FUNCTION( mrealgorithmspy, triangulate_point_cloud, MRE::triangulatePointCloud, "Creates mesh from given point cloud" )




