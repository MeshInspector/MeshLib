#include "TestMacros.h"
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMakeSphereMesh.h>
#include <MRMeshC/MRVDBConversions.h>

void testVDBConversions( void )
{
    MRSphereParams params;
    params.radius = 1.0f;
    params.numMeshVertices = 3000;
    
    MRMesh* mesh = mrMakeSphere( &params );

    MRMeshToVolumeSettings settings = mrVdbConversionsMeshToVolumeSettingsNew();
    settings.voxelSize = mrVector3fDiagonal( 0.1f );

    MRVdbVolume volume = mrVdbConversionsMeshToVolume( mesh, &settings, NULL );
    TEST_ASSERT( volume.min > -0.001f && volume.min < 0.001f );
    TEST_ASSERT( volume.max > 2.999f && volume.max < 3.001f );
    TEST_ASSERT( volume.dims.x == 26 );
    TEST_ASSERT( volume.dims.y == 26 );
    TEST_ASSERT( volume.dims.z == 26 );

    mrMeshFree( mesh );

    MRGridToMeshSettings gridToMeshSettings = mrVdbConversionsGridToMeshSettingsNew();
    gridToMeshSettings.voxelSize = mrVector3fDiagonal( 0.1f );
    gridToMeshSettings.isoValue = 1;

    MRMesh* restored = mrVdbConversionsGridToMesh( volume.data, &gridToMeshSettings, NULL );
    MRBox3f bbox = mrMeshComputeBoundingBox( restored, NULL );
    TEST_ASSERT( bbox.min.x > 0.199f && bbox.min.x < 0.201f );
    TEST_ASSERT( bbox.min.y > 0.199f && bbox.min.y < 0.201f );
    TEST_ASSERT( bbox.min.z > 0.199f && bbox.min.z < 0.201f );
    TEST_ASSERT( bbox.max.x > 2.394f && bbox.max.x < 2.396f );
    TEST_ASSERT( bbox.max.y > 2.394f && bbox.max.y < 2.396f );
    TEST_ASSERT( bbox.max.z > 2.394f && bbox.max.z < 2.396f );

    size_t pointsNum = mrMeshPointsNum( restored );
    TEST_ASSERT( pointsNum == 3748 );

    mrMeshFree( restored );
}