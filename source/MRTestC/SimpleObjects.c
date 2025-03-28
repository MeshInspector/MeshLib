#include "TestMacros.h"
#include "MRMeshC/MRCube.h"

#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRVector3.h"
#include "MRMeshC/MRCylinder.h"
#include "MRMeshC/MRTorus.h"
#include "MRMeshC/MRMakeSphereMesh.h"

void testMrMakeCube(void) {
    const MRVector3f size = {1.0f, 1.0f, 1.0f};
    const MRVector3f base = {0.0f, 0.0f, 0.0f};
    MRMesh *cube = mrMakeCube(&size, &base);

    const double area = mrMeshArea( cube, NULL );
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 6.0f, 0.001f );

    mrMeshFree(cube);
}


void testMrMakeCylinderAdvancedParametersNew(void) {
    const MRMakeCylinderAdvancedParameters params = mrMakeCylinderAdvancedParametersNew();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.radius0, 0.1f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.radius1, 0.1f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.startAngle, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.arcSize, 2.0f * 3.1415, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.length, 1.0f, 0.001f);
    TEST_ASSERT_INT_EQUAL(params.resolution, 16);
}

void testMrMakeCylinderAdvanced(void) {
    const MRMakeCylinderAdvancedParameters params = mrMakeCylinderAdvancedParametersNew();

    MRMesh *cylinder = mrMakeCylinderAdvanced(&params);
    TEST_ASSERT(cylinder != NULL);

    const double surfaceArea = mrMeshArea(cylinder, NULL);
    const double theoreticalSurfaceArea = 2.0 * 3.1415 * params.radius0 * params.length +
                                    3.1415 * params.radius0 * params.radius0 * 2.0; // assuming circular ends.
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, theoreticalSurfaceArea, 0.1f);

    mrMeshFree(cylinder);
}


void testMrMakeTorusParametersNew(void) {
    const MRMakeTorusParameters params = mrMakeTorusParametersNew();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.primaryRadius, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(params.secondaryRadius, 0.1f, 0.001f);
    TEST_ASSERT_INT_EQUAL(params.primaryResolution, 16);
    TEST_ASSERT_INT_EQUAL(params.secondaryResolution, 16);
}


void testMrMakeTorus(void) {
    MRMakeTorusParameters params = mrMakeTorusParametersNew();

    params.primaryRadius = 2.0f;
    params.secondaryRadius = 0.5f;
    params.primaryResolution = 32;
    params.secondaryResolution = 32;

    MRMesh *torus = mrMakeTorus(&params);
    TEST_ASSERT(torus != NULL);

    const double surfaceArea = mrMeshArea(torus, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, 39.257, 0.1f);

    mrMeshFree(torus);
}

void testMrMakeTorusWithSelfIntersections(void) {
    MRMakeTorusParameters params = mrMakeTorusParametersNew();

    params.primaryRadius = 2.0f;
    params.secondaryRadius = 0.5f;
    params.primaryResolution = 16;
    params.secondaryResolution = 16;

    MRMesh *torus = mrMakeTorusWithSelfIntersections(&params);
    TEST_ASSERT(torus != NULL);

    const double surfaceArea = mrMeshArea(torus, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(surfaceArea, 57.583, 0.01f);

    mrMeshFree(torus);
}


void testMrMakeSphere(void) {
    MRSphereParams params = mrSphereParamsNew();
    params.radius = 1.0f; // Set radius for the test
    params.numMeshVertices = 100; // Example value for testing

    MRMesh *sphere = mrMakeSphere(&params);
    TEST_ASSERT(sphere != NULL);

    const double area = mrMeshArea(sphere, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 12.157, 0.1f);

    mrMeshFree(sphere);
}

void testMrMakeUVSphere(void) {
    MRMakeUVSphereParameters params = mrMakeUvSphereParametersNew();
    params.radius = 1.5f; // Set radius for the test
    params.horizontalResolution = 32; // Increase horizontal resolution
    params.verticalResolution = 32; // Increase vertical resolution

    MRMesh *uvSphere = mrMakeUVSphere(&params);
    TEST_ASSERT(uvSphere != NULL);

    const double area = mrMeshArea(uvSphere, NULL);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(area, 28.15, 0.1f);

    mrMeshFree(uvSphere);
}
