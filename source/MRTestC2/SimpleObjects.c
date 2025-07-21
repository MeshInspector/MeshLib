#include "TestMacros.h"
#include "MRCMesh/MRCube.h"

#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRVector3.h"
#include "MRCMesh/MRCylinder.h"
#include "MRCMesh/MRTorus.h"
#include "MRCMesh/MRMakeSphereMesh.h"

void testMrMakeCube(void) {
    const MR_Vector3f size = {1.0f, 1.0f, 1.0f};
    const MR_Vector3f base = {0.0f, 0.0f, 0.0f};
    MR_Mesh *cube = MR_makeCube(&size, &base);

    const double area = MR_Mesh_area_const_MR_FaceBitSet_ptr( cube, NULL );
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 6.0f, 0.001f );

    MR_Mesh_Destroy(cube);
}

void testMrMakeCylinderAdvanced(void) {
    float radius0 = 1.0f;
    float radius1 = 0.5f;
    float startAngle = 0.0f;
    float arcSize = 2.0f * 3.1415f;
    float length = 1.0f;
    int32_t resolution = 16;
    MR_Mesh *cylinder = MR_makeCylinderAdvanced(&radius0, &radius1, &startAngle, &arcSize, &length, &resolution);

    TEST_ASSERT(cylinder != NULL);

    const double surfaceArea = MR_Mesh_area_const_MR_FaceBitSet_ptr(cylinder, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, 10.541437, 0.1f);

    MR_Mesh_Destroy(cylinder);
}


void testMrMakeTorus(void) {
    float primaryRadius = 2.0f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 32;
    int32_t secondaryResolution = 32;
    MR_Mesh* torus = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    TEST_ASSERT(torus != NULL);

    const double surfaceArea = MR_Mesh_area_const_MR_FaceBitSet_ptr(torus, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, 39.257, 0.1f);

    MR_Mesh_Destroy(torus);
}

void testMrMakeTorusWithSelfIntersections(void) {
    float primaryRadius = 2.0f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 16;
    int32_t secondaryResolution = 16;
    MR_Mesh* torus = MR_makeTorusWithSelfIntersections(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    TEST_ASSERT(torus != NULL);

    const double surfaceArea = MR_Mesh_area_const_MR_FaceBitSet_ptr(torus, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, 57.583, 0.01f);

    MR_Mesh_Destroy(torus);
}


void testMrMakeSphere(void) {
    MR_SphereParams* params = MR_SphereParams_DefaultConstruct();
    MR_SphereParams_Set_radius(params, 1.0f); // Set radius for the test
    MR_SphereParams_Set_numMeshVertices(params, 100); // Example value for testing

    MR_Mesh *sphere = MR_makeSphere(params);
    MR_SphereParams_Destroy(params);

    TEST_ASSERT(sphere != NULL);

    const double area = MR_Mesh_area_const_MR_FaceBitSet_ptr(sphere, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 12.157, 0.1f);

    MR_Mesh_Destroy(sphere);
}

void testMrMakeUVSphere(void) {
    float radius = 1.5f; // Set radius for the test
    int32_t horizontalResolution = 32; // Increase horizontal resolution
    int32_t verticalResolution = 32; // Increase vertical resolution

    MR_Mesh *uvSphere = MR_makeUVSphere(&radius, &horizontalResolution, &verticalResolution);
    TEST_ASSERT(uvSphere != NULL);

    const double area = MR_Mesh_area_const_MR_FaceBitSet_ptr(uvSphere, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 28.15, 0.1f);

    MR_Mesh_Destroy(uvSphere);
}
