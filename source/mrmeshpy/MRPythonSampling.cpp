#include "MRPython/MRPython.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRGridSampling.h"
#include "MRMesh/MRUniformSampling.h"
#include "MRMesh/MRIterativeSampling.h"
#include "MRMesh/MRPointCloud.h"
#include <pybind11/functional.h>

namespace MR
{

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

    m.def( "pointIterativeSampling", []( const PointCloud & pc, int n, ProgressCallback cb )
        {
            VertBitSet res;
            if ( auto x = pointIterativeSampling( pc, n, cb ) )
                res = std::move( *x );
            return res;
        }, pybind11::arg( "cloud" ), pybind11::arg( "numSamples" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "performs sampling of cloud points by iteratively removing one point with minimal metric (describing distance to the closest point and previous nearby removals), "
        "thus allowing stopping at any given number of samples" );
} )

} //namespace MR
