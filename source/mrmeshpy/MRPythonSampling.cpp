#include "MRMesh/MRPython.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRGridSampling.h"
#include "MRMesh/MRUniformSampling.h"
#include <pybind11/functional.h>

using namespace MR;

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointsSampling, [] ( pybind11::module_& m )
{
    m.def( "pointGridSampling", []( const PointCloud & pc, float d, ProgressCallback cb )
        {
            VertBitSet res;
            if ( auto x = pointGridSampling( pc, d, cb ) )
                res = std::move( *x );
            return res;
        }, pybind11::arg( "cloud" ), pybind11::arg( "voxelSize" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "performs sampling of point cloud vertices;\n"
        "subdivides point cloud bounding box on voxels of approximately given size and returns at most one vertex per voxel" );

    m.def( "pointUniformSampling", []( const PointCloud & pc, float d, ProgressCallback cb )
        {
            VertBitSet res;
            const UniformSamplingSettings s{ .distance = d, .progress = cb };
            if ( auto x = pointUniformSampling( pc, s ) )
                res = std::move( *x );
            return res;
        }, pybind11::arg( "pointCloud" ), pybind11::arg( "distance" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "Sample vertices, removing ones that are too close" );
} )
