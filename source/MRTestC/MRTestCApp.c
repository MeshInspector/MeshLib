#include "MRMeshBoolean.h"
#include "MRMeshDecimate.h"
#include "MRMeshFillHole.h"

int main( void )
{
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testMeshDecimate();
    testMeshFillHole();
}
