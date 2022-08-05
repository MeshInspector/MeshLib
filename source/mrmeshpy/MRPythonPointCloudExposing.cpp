#include "mrmeshpy/MRPython.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRBox.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointCloud, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::PointCloud>( m, "PointCloud" ).
        def( pybind11::init<>() ).
        def_readwrite( "points", &MR::PointCloud::points ).
        def_readwrite( "normals", &MR::PointCloud::normals ).
        def_readwrite( "validPoints", &MR::PointCloud::validPoints ).
        def( "getBoundingBox", &MR::PointCloud::getBoundingBox ).
        def( "invalidateCaches", &MR::PointCloud::invalidateCaches );
} )

