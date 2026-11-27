#include <MRCuda/MRCudaPointsProject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTorus.h>

#include <gtest/gtest.h>

using namespace MR;

#define ASSERT_EXPECTED( expr ) if ( auto res = ( expr ); !res ) { ASSERT_TRUE( ( (void)#expr, res ) ) << "Unexpected value: " << res.error(); }

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
        ASSERT_EXPECTED( cpuProjector.setPointCloud( torusPoints ) );
        ASSERT_EXPECTED( cpuProjector.findProjections( cpuResults, cube.points, {} ) );
    }
    ASSERT_EQ( cpuResults.size(), cube.points.size() );
    for ( auto i = 0; i < cube.points.size(); ++i )
        ASSERT_LT( cpuResults[i].distSq, FLT_MAX ) << "Incorrect result at index " << i;

    std::vector<PointsProjectionResult> cudaResults;
    {
        Cuda::PointsProjector cudaProjector;
        ASSERT_EXPECTED( cudaProjector.setPointCloud( torusPoints ) );
        ASSERT_EXPECTED( cudaProjector.findProjections( cudaResults, cube.points, {} ) );
    }
    ASSERT_EQ( cudaResults.size(), cube.points.size() );
    for ( auto i = 0; i < cube.points.size(); ++i )
    {
        ASSERT_NEAR( cudaResults[i].distSq, cpuResults[i].distSq, 1e-6f ) << "Incorrect result at index " << i;
        ASSERT_EQ( cudaResults[i].vId, cpuResults[i].vId ) << "Incorrect result at index " << i;
    }
}

TEST( MRCuda, PointsProjectorSkipSameIndex )
{
    // the order of points in the tree differs from the order of their ids
    PointCloud torusPoints;
    torusPoints.points = makeTorus().points;
    torusPoints.validPoints.resize( torusPoints.points.size(), true );
    torusPoints.invalidateCaches();

    FindProjectionOnPointsSettings settings;
    settings.skipSameIndex = true;

    std::vector<PointsProjectionResult> cpuResults;
    {
        PointsProjector cpuProjector;
        ASSERT_EXPECTED( cpuProjector.setPointCloud( torusPoints ) );
        ASSERT_EXPECTED( cpuProjector.findProjections( cpuResults, torusPoints.points, settings ) );
    }

    std::vector<PointsProjectionResult> cudaResults;
    {
        Cuda::PointsProjector cudaProjector;
        ASSERT_EXPECTED( cudaProjector.setPointCloud( torusPoints ) );
        ASSERT_EXPECTED( cudaProjector.findProjections( cudaResults, torusPoints.points, settings ) );
    }
    ASSERT_EQ( cudaResults.size(), torusPoints.points.size() );
    ASSERT_EQ( cudaResults.size(), cpuResults.size() );
    for ( auto i = 0; i < cudaResults.size(); ++i )
    {
        ASSERT_TRUE( cudaResults[i].vId.valid() ) << "No result at index " << i;
        ASSERT_NE( (int)cudaResults[i].vId, i ) << "The point is projected on itself at index " << i;
        // another point at the same distance is as good
        ASSERT_NEAR( cudaResults[i].distSq, cpuResults[i].distSq, 1e-6f ) << "Incorrect result at index " << i;
    }
}
