#include "TestMacros.h"

#include "MRMeshBuilder.h"

#include <MRMeshC/MRMakeSphereMesh.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshBuilder.h>

void testUniteCloseVertices( void )
{
    MRSphereParams params;
    params.radius = 1.0f;
    params.numMeshVertices = 3000;

    MRMesh* mesh = mrMakeSphere( &params );
    MRVertMap* vertMap = mrVertMapNew();

    int unitedCount = mrMeshBuilderUniteCloseVertices( mesh, 0.1f, false, vertMap );
    TEST_ASSERT( unitedCount == 2230 );
    TEST_ASSERT( vertMap->data[1000].id == 42 );

    mrVertMapFree( vertMap );
    mrMeshFree( mesh );
}
