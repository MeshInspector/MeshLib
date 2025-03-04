#include <MRMesh/MRSystem.h>

#include <gtest/gtest.h>

int main( int argc, char** argv )
{
    MR::setupLoggerByDefault();

    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
