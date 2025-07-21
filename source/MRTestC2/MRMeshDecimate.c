#include "TestMacros.h"

#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRCylinder.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshDecimate.h>
#include <MRCMesh/MRMeshTopology.h>

#define PI_F 3.14159265358979f

void testMeshDecimate( void )
{
    float radius0 = 0.5f;
    float radius1 = 0.5f;
    float startAngle = 0.0f;
    float arcSize = 20.0f / 180.0f * PI_F;
    float length = 1.0f;
    int32_t resolution = 16;
    MR_Mesh* meshCylinder = MR_makeCylinderAdvanced( &radius0, &radius1, &startAngle, &arcSize, &length, &resolution );

    // select all faces
    MR_FaceBitSet* regionForDecimation = MR_FaceBitSet_ConstructFromAnother( MR_PassBy_Copy, (MR_FaceBitSet *)MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( meshCylinder ) ) );
    MR_FaceBitSet* regionSaved = MR_FaceBitSet_ConstructFromAnother( MR_PassBy_Copy, regionForDecimation );

    // setup and run decimator
    MR_DecimateSettings* decimateSettings = MR_DecimateSettings_DefaultConstruct();
    MR_DecimateSettings_Set_maxError( decimateSettings, 0.001f );
    MR_DecimateSettings_Set_region( decimateSettings, regionForDecimation );
    MR_DecimateSettings_Set_maxTriangleAspectRatio( decimateSettings, 80.0f );

    MR_DecimateResult* decimateResults = MR_decimateMesh( meshCylinder, decimateSettings );
    MR_DecimateSettings_Destroy( decimateSettings );

    // compare regions and deleted vertices and faces
    TEST_ASSERT( !MR_equal_MR_BitSet( MR_FaceBitSet_UpcastTo_MR_BitSet( regionSaved ), MR_FaceBitSet_UpcastTo_MR_BitSet( regionForDecimation ) ) )
    TEST_ASSERT( *MR_DecimateResult_Get_vertsDeleted( decimateResults ) > 0 )
    TEST_ASSERT( *MR_DecimateResult_Get_facesDeleted( decimateResults ) > 0 )

    MR_FaceBitSet_Destroy( regionSaved );
    MR_FaceBitSet_Destroy( regionForDecimation );
    MR_Mesh_Destroy( meshCylinder );
}
