#include "MRMeshBoolean.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshDecimate.h"
#include "MRMeshFillHole.h"
#include "MRMeshNormals.h"

int main( void )
{
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testMeshCollidePrecise();
    testMeshDecimate();
    testMeshFillHole();
    testMeshNormals();
}
