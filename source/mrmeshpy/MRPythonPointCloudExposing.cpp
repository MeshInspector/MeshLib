#include "MRMesh/MRPython.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointCloudTriangulation.h"
#include "MRMesh/MRMeshToPointCloud.h"
#include "MRMesh/MRBox.h"
#include <pybind11/stl.h>
#include <pybind11/functional.h>

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PointCloud, MR::PointCloud )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointCloud, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( PointCloud ).
        def( pybind11::init<>() ).
        def_readwrite( "points", &MR::PointCloud::points ).
        def_readwrite( "normals", &MR::PointCloud::normals ).
        def_readwrite( "validPoints", &MR::PointCloud::validPoints ).
        def( "getBoundingBox", &MR::PointCloud::getBoundingBox, "returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())" ).
        def( "invalidateCaches", &MR::PointCloud::invalidateCaches, "Invalidates caches (e.g. aabb-tree) after a change in point cloud" );

    pybind11::class_<MR::TriangulationParameters>( m, "TriangulationParameters", "Parameters of point cloud triangulation" ).
        def( pybind11::init<>() ).
        def_readwrite( "numNeighbours", &MR::TriangulationParameters::numNeighbours,
            "The number of nearest neighbor points to use for building of local triangulation.\n"
            "note: Too small value can make not optimal triangulation and additional holes\n"
            "Too big value increases difficulty of optimization and decreases performance" ).
        def_readwrite( "critAngle", &MR::TriangulationParameters::critAngle, "Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)" ).
        def_readwrite( "critHoleLength", &MR::TriangulationParameters::critHoleLength,
            "Critical length of hole (all holes with length less then this value will be filled)\n"
            "If value is subzero it is set automaticly to 0.7*bbox.diagonal()" );

    m.def( "triangulatePointCloud", &MR::triangulatePointCloud,
        pybind11::arg( "pointCloud" ), pybind11::arg_v( "params", MR::TriangulationParameters(), "TriangulationParameters()" ), pybind11::arg( "progressCb" ) = MR::ProgressCallback{},
        "Creates mesh from given point cloud according params\n"
        "Returns empty optional if was interrupted by progress bar" );

    m.def( "meshToPointCloud", &MR::meshToPointCloud,
        pybind11::arg( "mesh" ), pybind11::arg( "saveNormals" ) = true, pybind11::arg( "verts" ) = nullptr,
        "Mesh to PointCloud" );
} )
