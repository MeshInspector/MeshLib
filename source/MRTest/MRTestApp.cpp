#include <gtest/gtest.h>
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRCommandLoop.h"

int main( int argc, char** argv )
{
    MR::setupLoggerByDefault();

    ::testing::InitGoogleTest(&argc, argv);
    MR::CommandLoop::removeCommands( false ); // that are added there by plugin constructors
    return RUN_ALL_TESTS();
}
