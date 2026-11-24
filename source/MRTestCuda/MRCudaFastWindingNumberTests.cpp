#include <MRCuda/MRCudaFastWindingNumber.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRDistanceToMeshOptions.h>
#include <MRMesh/MRFastWindingNumber.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRVector.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace MR;

namespace
{

AffineXf3f gridToMesh( const Box3f& box, const Vector3i& dims )
{
    const auto size = box.size();
    return AffineXf3f( Matrix3f::scale( size.x / ( dims.x - 1 ), size.y / ( dims.y - 1 ), size.z / ( dims.z - 1 ) ), box.min );
}

} // namespace

// the output vectors are empty on input and must be resized by the calls
TEST( MRCuda, FastWindingNumberFromGrid )
{
    const auto torus = makeTorus( 1.0f, 0.3f, 32, 16 );
    const Vector3i dims{ 24, 24, 12 };
    auto box = torus.computeBoundingBox();
    box.include( box.min - Vector3f::diagonal( 0.1f ) );
    box.include( box.max + Vector3f::diagonal( 0.1f ) );
    const auto xf = gridToMesh( box, dims );
    const auto numVoxels = size_t( dims.x ) * dims.y * dims.z;

    FastWindingNumber cpu( torus );
    Cuda::FastWindingNumber gpu( torus );

    std::vector<float> cpuWn, gpuWn;
    ASSERT_TRUE( cpu.calcFromGrid( cpuWn, dims, xf, 2, {} ) );
    auto r = gpu.calcFromGrid( gpuWn, dims, xf, 2, {} );
    ASSERT_TRUE( r ) << r.error();
    ASSERT_EQ( gpuWn.size(), numVoxels );
    for ( size_t i = 0; i < numVoxels; ++i )
        ASSERT_NEAR( gpuWn[i], cpuWn[i], 1e-3f ) << "voxel " << i;

    DistanceToMeshOptions opts;
    opts.maxDistSq = 0.2f * 0.2f;
    VoxelScalars cpuDist, gpuDist;
    ASSERT_TRUE( cpu.calcFromGridWithDistances( cpuDist, dims, xf, opts, {} ) );
    r = gpu.calcFromGridWithDistances( gpuDist, dims, xf, opts, {} );
    ASSERT_TRUE( r ) << r.error();
    ASSERT_EQ( gpuDist.size(), numVoxels );
    for ( size_t i = 0; i < numVoxels; ++i )
    {
        const float g = gpuDist.vec[i], c = cpuDist.vec[i];
        ASSERT_EQ( std::isnan( g ), std::isnan( c ) ) << "voxel " << i;
        if ( !std::isnan( c ) )
        {
            ASSERT_NEAR( g, c, 1e-5f ) << "voxel " << i;
        }
    }
}
