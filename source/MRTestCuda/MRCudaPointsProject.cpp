#include <MRCuda/MRCudaPointsProject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTorus.h>

#include <gtest/gtest.h>

using namespace MR;

#define ASSERT_EXPECTED( expr ) if ( auto res = ( expr ); !res ) ASSERT_TRUE( ( (void)#expr, res ) ) << "Unexpected value: " << res.error()

TEST( MRCuda, PointsProjector )
{
    const auto cube = makeCube();
    const auto torus = makeTorus();

    PointCloud torusPoints;
    torusPoints.points = torus.points;
    torusPoints.validPoints.resize( torus.points.size(), true );
    torusPoints.invalidateCaches();

    std::vector<PointsProjectionResult> cpuResults;
    {
        PointsProjector cpuProjector;
        cpuProjector.setPointCloud( torusPoints );
        cpuProjector.findProjections( cpuResults, cube.points.vec_, {} );
    }
    ASSERT_EQ( cpuResults.size(), cube.points.size() );
    for ( auto i = 0; i < cube.points.size(); ++i )
        ASSERT_LT( cpuResults[i].distSq, FLT_MAX ) << "Incorrect result at index " << i;

    std::vector<PointsProjectionResult> cudaResults;
    {
        Cuda::PointsProjector cudaProjector;
        ASSERT_EXPECTED( cudaProjector.setPointCloud( torusPoints ) );
        ASSERT_EXPECTED( cudaProjector.findProjections( cudaResults, cube.points.vec_, {} ) );
    }
    ASSERT_EQ( cudaResults.size(), cube.points.size() );
    for ( auto i = 0; i < cube.points.size(); ++i )
    {
        ASSERT_NEAR( cudaResults[i].distSq, cpuResults[i].distSq, 1e-6f ) << "Incorrect result at index " << i;
        ASSERT_EQ( cudaResults[i].vId, cpuResults[i].vId ) << "Incorrect result at index " << i;
    }
}
