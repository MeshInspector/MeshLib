#include "TestMacros.h"
#include "MRCMesh/MRTorus.h"
#include "MRCMesh/MRMeshPart.h"
#include "MRCMesh/MRMeshCollide.h"
#include "MRCMesh/MRMesh.h"

void testMeshCollide( void )
{
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 8;
    int32_t secondaryResolution = 8;
    MR_Mesh* meshA = MR_makeTorus( &primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL );

    secondaryRadius = 0.2f;
    MR_Mesh* meshB = MR_makeTorus( &primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL );

    MR_MeshPart* meshAPart = MR_MeshPart_Construct( meshA, NULL );
    MR_MeshPart* meshBPart = MR_MeshPart_Construct( meshB, NULL );

    TEST_ASSERT( MR_isInside_MR_MeshPart( meshBPart, meshAPart, NULL ) );
    TEST_ASSERT( !MR_isInside_MR_MeshPart( meshAPart, meshBPart, NULL ) );

    MR_MeshPart_Destroy( meshAPart );
    MR_MeshPart_Destroy( meshBPart );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}
