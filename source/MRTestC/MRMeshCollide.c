#include "TestMacros.h"
#include "MRMeshC/MRTorus.h"
#include "MRMeshC/MRMeshPart.h"
#include "MRMeshC/MRMeshCollide.h"
#include "MRMeshC/MRMesh.h"

void testMeshCollide( void )
{
    MRMakeTorusParameters params = {
    .primaryRadius = 1.1f,
    .secondaryRadius = 0.5f,
    .primaryResolution = 8,
    .secondaryResolution = 8,
    };
    MRMesh* meshA = mrMakeTorus( &params );
    params.secondaryRadius = 0.2f;
    MRMesh* meshB = mrMakeTorus( &params );

    MRMeshPart meshAPart = { meshA, NULL };
    MRMeshPart meshBPart = { meshB, NULL };

    TEST_ASSERT( mrIsInside( &meshBPart, &meshAPart, NULL ) );
    TEST_ASSERT( !mrIsInside( &meshAPart, &meshBPart, NULL ) );

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}
