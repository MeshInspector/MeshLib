#include <MRCuda/MRCudaBasic.h>
#include <MRMesh/MRSystem.h>

#include <gtest/gtest.h>

int main( int argc, char** argv )
{
    MR::setupLoggerByDefault();

    int driverVersion, runtimeVersion, computeMajor, computeMinor;
    if ( !MR::Cuda::isCudaAvailable( &driverVersion, &runtimeVersion, &computeMajor, &computeMinor ) )
    {
        spdlog::critical( "No CUDA-capable device found" );
        return EXIT_FAILURE;
    }
    spdlog::info( "Driver version: {}.{}", driverVersion / 1000, ( driverVersion % 1000 ) / 10 );
    spdlog::info( "Runtime version: {}.{}", runtimeVersion / 1000, ( runtimeVersion % 1000 ) / 10 );
    spdlog::info( "Compute version: {}.{}", computeMajor, computeMinor );

    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
