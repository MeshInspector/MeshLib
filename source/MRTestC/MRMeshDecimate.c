#include "TestMacros.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRCylinder.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshDecimate.h>
#include <MRMeshC/MRMeshTopology.h>

#define PI_F 3.14159265358979f

void testMeshDecimate( void )
{
    MRMakeCylinderAdvancedParameters params = {
        .radius0 = 0.5f,
        .radius1 = 0.5f,
        .startAngle = 0.0f,
        .arcSize = 20.0f / 180.0f * PI_F,
        .length = 1.0f,
        .resolution = 16
    };
    MRMesh* meshCylinder = mrMakeCylinderAdvanced( &params );

    // select all faces
    MRFaceBitSet* regionForDecimation = mrFaceBitSetCopy( mrMeshTopologyGetValidFaces( mrMeshTopology( meshCylinder ) ) );
    MRFaceBitSet* regionSaved = mrFaceBitSetCopy( regionForDecimation );

    // setup and run decimator
    MRDecimateSettings decimateSettings = mrDecimateSettingsNew();
    decimateSettings.maxError = 0.001f;
    decimateSettings.region = regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    MRDecimateResult decimateResults = mrDecimateMesh( meshCylinder, &decimateSettings );

    // compare regions and deleted vertices and faces
    TEST_ASSERT( !mrBitSetEq( (MRBitSet*)regionSaved, (MRBitSet*)regionForDecimation ) )
    TEST_ASSERT( decimateResults.vertsDeleted > 0 )
    TEST_ASSERT( decimateResults.facesDeleted > 0 )

    mrFaceBitSetFree( regionSaved );
    mrFaceBitSetFree( regionForDecimation );
    mrMeshFree( meshCylinder );
}
