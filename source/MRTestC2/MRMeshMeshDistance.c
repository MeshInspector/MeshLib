#include "MRMeshMeshDistance.h"
#include "TestMacros.h"

#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMeshMeshDistance.h>
#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRMesh.h>

#include <float.h>

void testMeshMeshDistance( void )
{
    float radius = 1.f;
    int32_t horizontalResolution = 8;
    int32_t verticalResolution = 8;

    MR_Mesh* sphere1 = MR_makeUVSphere( &radius, &horizontalResolution, &verticalResolution );

    MR_MeshPart* wholeSphere1 = MR_MeshPart_Construct( sphere1, NULL );
    MR_MeshMeshDistanceResult* d11 = MR_findDistance( wholeSphere1, wholeSphere1, NULL, &(float){FLT_MAX} );
    TEST_ASSERT( *MR_MeshMeshDistanceResult_Get_distSq( d11 ) == 0 );

    const MR_Vector3f translation = { 0.0f, 0.0f, 3.0f };
    const MR_AffineXf3f zShift = MR_AffineXf3f_translation( &translation );

    MR_MeshMeshDistanceResult* d1z = MR_findDistance( wholeSphere1, wholeSphere1, &zShift, &(float){FLT_MAX} );
    TEST_ASSERT( *MR_MeshMeshDistanceResult_Get_distSq( d1z ) == 1 );

    radius = 2.0f;
    MR_Mesh* sphere2 = MR_makeUVSphere( &radius, &horizontalResolution, &verticalResolution );

    MR_MeshPart* wholeSphere2 = MR_MeshPart_Construct( sphere2, NULL );
    MR_MeshMeshDistanceResult* d12 = MR_findDistance( wholeSphere1, wholeSphere2, NULL, &(float){FLT_MAX} );
    float dist12 = sqrtf( *MR_MeshMeshDistanceResult_Get_distSq( d12 ) );
    TEST_ASSERT( dist12 > 0.9f && dist12 < 1.0f );

    MR_MeshMeshDistanceResult_Destroy( d12 );
    MR_MeshPart_Destroy( wholeSphere2 );
    MR_Mesh_Destroy( sphere2 );
    MR_MeshMeshDistanceResult_Destroy( d1z );
    MR_MeshMeshDistanceResult_Destroy( d11 );
    MR_MeshPart_Destroy( wholeSphere1 );
    MR_Mesh_Destroy( sphere1 );
}
