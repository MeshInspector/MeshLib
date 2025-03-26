#include "TestMacros.h"
#include "MRMeshC/MRAffineXf.h"


/// Test function for mrAffineXf3fNew
void testMrAffineXf3fNew() {
    const MRAffineXf3f result = mrAffineXf3fNew();

    // Check if the matrix A is an identity matrix
    const MRMatrix3f identity = mrMatrix3fIdentity();
    TEST_ASSERT(mrMatrix3fEqual(&result.A, &identity));

    // Check if the translation vector b is zero
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 0.0f, 0.0001f);
}

/// Test function for mrAffineXf3fTranslation
void testMrAffineXf3fTranslation() {
    const MRVector3f translation = {1.0f, 2.0f, 3.0f};
    const MRAffineXf3f result = mrAffineXf3fTranslation(&translation);

    // Check if the matrix A is still an identity matrix
    const MRMatrix3f identity = mrMatrix3fIdentity();
    TEST_ASSERT(mrMatrix3fEqual(&result.A, &identity));

    // Check if the translation vector b has the correct value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 3.0f, 0.0001f);
}

/// Test function for mrAffineXf3fLinear
void testMrAffineXf3fLinear() {
    const MRMatrix3f linearMatrix = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    const MRAffineXf3f result = mrAffineXf3fLinear(&linearMatrix);

    // Check if the matrix A has the correct value
    TEST_ASSERT(mrMatrix3fEqual(&result.A, &linearMatrix));

    // Check if the translation vector b is still zero
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 0.0f, 0.0001f);
}

/// Test function for mrAffineXf3fMul
void testMrAffineXf3fMul() {
    const MRAffineXf3f xf1 = mrAffineXf3fTranslation(&(MRVector3f){1.0f, 2.0f, 3.0f});
    const MRAffineXf3f xf2 = mrAffineXf3fLinear(&(MRMatrix3f){{2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, {0.0f, 0.0f, 2.0f}});
    const MRAffineXf3f result = mrAffineXf3fMul(&xf1, &xf2);

    // Expected results
    const MRMatrix3f expectedA = mrMatrix3fMul(&xf1.A, &xf2.A);
    const MRVector3f expectedB = {
        xf1.A.x.x * xf2.b.x + xf1.A.y.x * xf2.b.y + xf1.A.z.x * xf2.b.z + xf1.b.x,
        xf1.A.x.y * xf2.b.x + xf1.A.y.y * xf2.b.y + xf1.A.z.y * xf2.b.z + xf1.b.y,
        xf1.A.x.z * xf2.b.x + xf1.A.y.z * xf2.b.y + xf1.A.z.z * xf2.b.z + xf1.b.z
    };

    // Check if the resulting matrix A matches the expected value
    TEST_ASSERT(mrMatrix3fEqual(&result.A, &expectedA));

    // Check if the resulting translation vector b matches the expected value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, expectedB.x, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, expectedB.y, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, expectedB.z, 0.0001f);
}

/// Test function for mrAffineXf3fApply
void testMrAffineXf3fApply() {
    const MRAffineXf3f xf = mrAffineXf3fTranslation(&(MRVector3f){1.0f, 2.0f, 3.0f});
    const MRVector3f point = {4.0f, 5.0f, 6.0f};
    const MRVector3f result = mrAffineXf3fApply(&xf, &point);

    // Expected result
    const MRVector3f expectedResult = {
        xf.A.x.x * point.x + xf.A.y.x * point.y + xf.A.z.x * point.z + xf.b.x,
        xf.A.x.y * point.x + xf.A.y.y * point.y + xf.A.z.y * point.z + xf.b.y,
        xf.A.x.z * point.x + xf.A.y.z * point.y + xf.A.z.z * point.z + xf.b.z
    };

    // Check if the resulting transformed point matches the expected value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, expectedResult.x, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, expectedResult.y, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, expectedResult.z, 0.0001f);
}
