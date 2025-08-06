#include "TestMacros.h"

#include "MRMeshMeshDistance.h"

#include "MRMeshC/MRMakeSphereMesh.h"
#include "MRMeshC/MRMeshMeshDistance.h"
#include "MRMeshC/MRAffineXf.h"
#include "MRMeshC/MRMesh.h"

#include <float.h>

void testMeshMeshDistance( void )
{
    MRMakeUVSphereParameters params = mrMakeUvSphereParametersNew();
    params.radius = 1.0f;
    params.horizontalResolution = 8;
    params.verticalResolution = 8;

    MRMesh* sphere1 = mrMakeUVSphere( &params );

    MRMeshPart wholeSphere1 = { sphere1, NULL };
    MRMeshMeshDistanceResult d11 = mrFindDistance( &wholeSphere1, &wholeSphere1, NULL, FLT_MAX );
    TEST_ASSERT( d11.distSq == 0 );

    const MRVector3f translation = { 0.0f, 0.0f, 3.0f };
    const MRAffineXf3f zShift = mrAffineXf3fTranslation( &translation );

    MRMeshMeshDistanceResult d1z = mrFindDistance( &wholeSphere1, &wholeSphere1, &zShift, FLT_MAX );
    TEST_ASSERT( d1z.distSq == 1 );

    params.radius = 2.0f;
    MRMesh* sphere2 = mrMakeUVSphere( &params );

    MRMeshPart wholeSphere2 = { sphere2, NULL };
    MRMeshMeshDistanceResult d12 = mrFindDistance( &wholeSphere1, &wholeSphere2, NULL, FLT_MAX );
    float dist12 = sqrtf( d12.distSq );
    TEST_ASSERT( dist12 > 0.9f && dist12 < 1.0f );

    mrMeshFree( sphere2 );
    mrMeshFree( sphere1 );
}
