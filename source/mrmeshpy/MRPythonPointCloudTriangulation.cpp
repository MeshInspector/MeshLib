#include "mrmeshpy/MRPython.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointCloudTriangulation.h"
#include <pybind11/stl.h>

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, TriangulationParameters, []( pybind11::module_& m )
{
    pybind11::class_<MR::TriangulationParameters>( m, "TriangulationParameters" ).
        def( pybind11::init<>() ).
        def_readwrite( "avgNumNeighbours", &MR::TriangulationParameters::avgNumNeighbours ).
        def_readwrite( "critAngle", &MR::TriangulationParameters::critAngle ).
        def_readwrite( "critHoleLength", &MR::TriangulationParameters::critHoleLength );

    m.def( "triangulate_point_cloud", [] ( const MR::PointCloud& pointCloud, const MR::TriangulationParameters& params )
    {
        return MR::triangulatePointCloud( pointCloud, params );
    },
        pybind11::arg( "pointCloud" ), pybind11::arg( "params" ) = MR::TriangulationParameters{},
        "Creates mesh from given point cloud" );
} )
