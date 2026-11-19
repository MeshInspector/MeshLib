#include <MRCuda/config.h>
#ifndef MRCUDA_NO_VOXELS
#include <MRCuda/MRCudaPointsToDistanceVolume.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRVoxels/MRPointsToDistanceVolume.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace MR;

namespace
{

// compares the volumes made on CPU and on GPU (as a whole and by parts)
void testPointsToDistanceVolume( const PointCloud& cloud, const PointsToDistanceVolumeParams& params )
{
    const auto cpuRes = MR::pointsToDistanceVolume( cloud, params );
    ASSERT_TRUE( cpuRes.has_value() ) << cpuRes.error();
    const auto& cpu = cpuRes->data;

    const auto cudaRes = Cuda::pointsToDistanceVolume( cloud, params );
    ASSERT_TRUE( cudaRes.has_value() ) << cudaRes.error();
    const auto& cuda = cudaRes->data;
    ASSERT_EQ( cuda.size(), cpu.size() );

    Vector<float, VoxelId> parts( cpu.size(), 0.f );
    const size_t layerSize = size_t( params.dimensions.x ) * params.dimensions.y;
    const auto partsRes = Cuda::pointsToDistanceVolumeByParts( cloud, params,
        [&] ( const SimpleVolumeMinMax& part, int zOffset ) -> Expected<void>
    {
        const size_t partSize = layerSize * part.dims.z;
        const size_t offset = layerSize * zOffset;
        if ( offset + partSize > parts.size() || partSize > part.data.size() )
            return unexpected( "invalid part" );
        std::copy_n( part.data.data(), partSize, parts.data() + offset );
        return {};
    }, 1 );
    ASSERT_TRUE( partsRes.has_value() ) << partsRes.error();

    // only the voxels having a value in all volumes are compared: the weight of a voxel can be just at minWeight;
    // and a point on the very border of the ball of a voxel may be taken or not, changing the value
    const float eps = 1e-4f * params.sigma;
    size_t numValid = 0, numCompared = 0, numDifferent = 0;
    for ( auto v = 0_vox; v < cpu.size(); ++v )
    {
        if ( std::isnan( cpu[v] ) )
            continue;
        ++numValid;
        if ( std::isnan( cuda[v] ) || std::isnan( parts[v] ) )
            continue;
        ++numCompared;
        if ( std::abs( cuda[v] - cpu[v] ) > eps || std::abs( parts[v] - cpu[v] ) > eps )
            ++numDifferent;
    }
    ASSERT_GT( numValid, cpu.size() / 100 );
    ASSERT_GE( numCompared, numValid - numValid / 1000 );
    ASSERT_LE( numDifferent, numValid / 1000 );
}

} // namespace

TEST( MRCuda, PointsToDistanceVolume )
{
    const auto sphere = makeUVSphere( 1.f, 32, 32 );
    PointCloud cloud;
    cloud.points = sphere.points;
    cloud.validPoints = sphere.topology.getValidVerts();
    cloud.normals.resize( cloud.points.size() );
    for ( auto v : cloud.validPoints )
        cloud.normals[v] = cloud.points[v].normalized();

    PointsToDistanceVolumeParams params;
    params.dimensions = Vector3i( 40, 40, 43 );
    params.origin = Vector3f::diagonal( -1.3f );
    params.voxelSize = Vector3f::diagonal( 2.6f / 40 );
    params.sigma = 0.2f;
    testPointsToDistanceVolume( cloud, params );

    params.sqrtAngleWeight = true;
    testPointsToDistanceVolume( cloud, params );

    params.sqrtAngleWeight = false;
    params.invSigmaModifier = 1.5f;
    testPointsToDistanceVolume( cloud, params );
}
#endif
