#include "MRBitSet.h"
#include "MRMeshBoolean.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshDecimate.h"
#include "MRMeshFillHole.h"
#include "MRMeshNormals.h"
#include "MRMeshComponents.h"
#include "MRMeshBuilder.h"

int main( void )
{
    testBitSet();
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testBooleanMapper();
    testMeshCollidePrecise();
    testMeshDecimate();
    testMeshFillHole();
    testMeshNormals();
    testComponentsMap();
    testLargeRegions();
    testUniteCloseVertices();
}
