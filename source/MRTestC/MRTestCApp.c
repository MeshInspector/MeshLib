#include "MRMeshBoolean.h"
#include "MRMeshDecimate.h"

int main( int argc, char* argv[] )
{
    (void)argc, (void)argv;
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testMeshDecimate();
    return 0;
}
