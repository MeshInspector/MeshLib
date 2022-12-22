#include "MRMesh/MRPython.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRGridSampling.h"
#include "MRMesh/MRUniformSampling.h"

using namespace MR;

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointsSampling, [] ( pybind11::module_& m )
{
    m.def( "pointGridSampling", ( VertBitSet( * )( const PointCloud&, float, ProgressCallback ) )&MR::pointGridSampling, pybind11::arg( "cloud" ), pybind11::arg( "voxelSize" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "performs sampling of point cloud vertices;\n"
        "subdivides point cloud bounding box on voxels of approximately given size and returns at most one vertex per voxel" );

    m.def( "pointUniformSampling", ( VertBitSet( * )( const PointCloud&, float, ProgressCallback ) )&MR::pointUniformSampling, pybind11::arg( "pointCloud" ), pybind11::arg( "distance" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "Sample vertices, removing ones that are too close" );
} )
