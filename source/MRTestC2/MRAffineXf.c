#include "TestMacros.h"
#include "MRCMesh/MRAffineXf.h"
#include "MRCMesh/MRMatrix3.h"


/// Test function for MR_AffineXf3f_DefaultConstruct
void testMrAffineXf3fNew(void) {
    MR_AffineXf3f result = MR_AffineXf3f_DefaultConstruct();

    // Check if the matrix A is an identity matrix
    MR_Matrix3f identity = MR_Matrix3f_identity();
    TEST_ASSERT(MR_equal_MR_Matrix3f(&result.A, &identity));

    // Check if the translation vector b is zero
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 0.0f, 0.0001f);
}

/// Test function for mrAffineXf3fTranslation
void testMrAffineXf3fTranslation(void) {
    const MR_Vector3f translation = {1.0f, 2.0f, 3.0f};
    const MR_AffineXf3f result = MR_AffineXf3f_translation(&translation);

    // Check if the matrix A is still an identity matrix
    const MR_Matrix3f identity = MR_Matrix3f_identity();
    TEST_ASSERT(MR_equal_MR_Matrix3f(&result.A, &identity));

    // Check if the translation vector b has the correct value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 3.0f, 0.0001f);
}

/// Test function for mrAffineXf3fLinear
void testMrAffineXf3fLinear(void) {
    const MR_Matrix3f linearMatrix = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    const MR_AffineXf3f result = MR_AffineXf3f_linear(&linearMatrix);

    // Check if the matrix A has the correct value
    TEST_ASSERT(MR_equal_MR_Matrix3f(&result.A, &linearMatrix));

    // Check if the translation vector b is still zero
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, 0.0f, 0.0001f);
}

/// Test function for mrAffineXf3fMul
void testMrAffineXf3fMul(void) {
    const MR_AffineXf3f xf1 = MR_AffineXf3f_translation(&(MR_Vector3f){1.0f, 2.0f, 3.0f});
    const MR_AffineXf3f xf2 = MR_AffineXf3f_linear(&(MR_Matrix3f){{2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, {0.0f, 0.0f, 2.0f}});
    const MR_AffineXf3f result = MR_mul_MR_AffineXf3f(&xf1, &xf2);

    // Expected results
    const MR_Matrix3f expectedA = MR_mul_MR_Matrix3f(&xf1.A, &xf2.A);
    const MR_Vector3f expectedB = {
        xf1.A.x.x * xf2.b.x + xf1.A.y.x * xf2.b.y + xf1.A.z.x * xf2.b.z + xf1.b.x,
        xf1.A.x.y * xf2.b.x + xf1.A.y.y * xf2.b.y + xf1.A.z.y * xf2.b.z + xf1.b.y,
        xf1.A.x.z * xf2.b.x + xf1.A.y.z * xf2.b.y + xf1.A.z.z * xf2.b.z + xf1.b.z
    };

    // Check if the resulting matrix A matches the expected value
    TEST_ASSERT(MR_equal_MR_Matrix3f(&result.A, &expectedA));

    // Check if the resulting translation vector b matches the expected value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.x, expectedB.x, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.y, expectedB.y, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.b.z, expectedB.z, 0.0001f);
}

/// Test function for mrAffineXf3fApply
void testMrAffineXf3fApply(void) {
    const MR_AffineXf3f xf = MR_AffineXf3f_translation(&(MR_Vector3f){1.0f, 2.0f, 3.0f});
    const MR_Vector3f point = {4.0f, 5.0f, 6.0f};
    const MR_Vector3f result = MR_AffineXf3f_call(&xf, &point);

    // Expected result
    const MR_Vector3f expectedResult = {
        xf.A.x.x * point.x + xf.A.y.x * point.y + xf.A.z.x * point.z + xf.b.x,
        xf.A.x.y * point.x + xf.A.y.y * point.y + xf.A.z.y * point.z + xf.b.y,
        xf.A.x.z * point.x + xf.A.y.z * point.y + xf.A.z.z * point.z + xf.b.z
    };

    // Check if the resulting transformed point matches the expected value
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, expectedResult.x, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, expectedResult.y, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, expectedResult.z, 0.0001f);
}
