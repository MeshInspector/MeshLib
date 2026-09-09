#include <MRCuda/MRCudaBasic.h>
#include <MRMesh/MRSystem.h>
#include <MRPch/MRSpdlog.h>

#include <gtest/gtest.h>

int main( int argc, char** argv )
{
    MR::setupLoggerByDefault();

    auto info = MR::Cuda::getDeviceInfo();
    if ( !info )
    {
        spdlog::critical( "CUDA error: {}", info.error() );
        return EXIT_FAILURE;
    }
    spdlog::info( "Driver version: {}.{}", info->driverVersion / 1000, ( info->driverVersion % 1000 ) / 10 );
    spdlog::info( "Runtime version: {}.{}", info->runtimeVersion / 1000, ( info->runtimeVersion % 1000 ) / 10 );
    spdlog::info( "Compute version: {}.{}", info->computeMajor, info->computeMinor );
    if ( !info->fitForComputations() )
    {
        spdlog::critical( "CUDA does not fit for computations" );
        return EXIT_FAILURE;
    }

    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
