#include "TestMacros.h"

#include "MRMeshBuilder.h"

#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBuilder.h>
#include <MRCMesh/MRVector.h>

void testUniteCloseVertices( void )
{
    MR_SphereParams* params = MR_SphereParams_DefaultConstruct();
    MR_SphereParams_Set_radius( params, 1.0f );
    MR_SphereParams_Set_numMeshVertices( params, 3000 );

    MR_Mesh* mesh = MR_makeSphere( params );
    MR_SphereParams_Destroy( params );

    MR_VertMap* vertMap = MR_VertMap_DefaultConstruct();

    int unitedCount = MR_MeshBuilder_uniteCloseVertices_4( mesh, 0.1f, &(bool){false}, vertMap );
    TEST_ASSERT( unitedCount == 2230 );
    TEST_ASSERT( MR_VertMap_index( vertMap, (MR_VertId){1000} )->id_ == 42 );

    MR_VertMap_Destroy( vertMap );
    MR_Mesh_Destroy( mesh );
}
