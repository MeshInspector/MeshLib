#include "TestMacros.h"
#include "MRCMesh/MRVector3.h"

void testMrVector3fDiagonal(void) {
    MR_Vector3f vec = MR_Vector3f_diagonal(3.0f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 3.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 3.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 3.0f, 0.0001f);
}

void testMrVector3fPlusX(void) {
    MR_Vector3f vec = MR_Vector3f_plusX();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 0.0f, 0.0001f);
}

void testMrVector3fPlusY(void) {
    MR_Vector3f vec = MR_Vector3f_plusY();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 0.0f, 0.0001f);
}

void testMrVector3fPlusZ(void) {
    MR_Vector3f vec = MR_Vector3f_plusZ();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 1.0f, 0.0001f);
}

void testMrVector3fAdd(void) {
    MR_Vector3f a = {1.0f, 2.0f, 3.0f};
    MR_Vector3f b = {4.0f, 5.0f, 6.0f};
    MR_Vector3f result = MR_add_MR_Vector3f(&a, &b);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 5.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 7.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 9.0f, 0.0001f);
}

void testMrVector3fSub(void) {
    MR_Vector3f a = {5.0f, 7.0f, 9.0f};
    MR_Vector3f b = {4.0f, 5.0f, 6.0f};
    MR_Vector3f result = MR_sub_MR_Vector3f(&a, &b);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 3.0f, 0.0001f);
}

void testMrVector3fMulScalar(void) {
    MR_Vector3f vec = {1.0f, 2.0f, 3.0f};
    float scalar = 2.0f;
    MR_Vector3f result = MR_mul_MR_Vector3f_float(&vec, scalar);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 4.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 6.0f, 0.0001f);
}

void testMrVector3fLengthSq(void) {
    MR_Vector3f vec = {1.0f, 2.0f, 3.0f};
    float lengthSq = MR_Vector3f_lengthSq(&vec);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(lengthSq, 14.0f, 0.0001f);
}

void testMrVector3fLength(void) {
    MR_Vector3f vec = {3.0f, 4.0f, 0.0f};
    float length = MR_Vector3f_length(&vec);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(length, 5.0f, 0.0001f);
}

void testMrVector3iDiagonal(void) {
    MR_Vector3i vec = MR_Vector3i_diagonal(3);

    TEST_ASSERT_INT_EQUAL(vec.x, 3);
    TEST_ASSERT_INT_EQUAL(vec.y, 3);
    TEST_ASSERT_INT_EQUAL(vec.z, 3);
}

void testMrVector3iPlusX(void) {
    MR_Vector3i vec = MR_Vector3i_plusX();

    TEST_ASSERT_INT_EQUAL(vec.x, 1);
    TEST_ASSERT_INT_EQUAL(vec.y, 0);
    TEST_ASSERT_INT_EQUAL(vec.z, 0);
}

void testMrVector3iPlusY(void) {
    MR_Vector3i vec = MR_Vector3i_plusY();

    TEST_ASSERT_INT_EQUAL(vec.x, 0);
    TEST_ASSERT_INT_EQUAL(vec.y, 1);
    TEST_ASSERT_INT_EQUAL(vec.z, 0);
}

void testMrVector3iPlusZ(void) {
    MR_Vector3i vec = MR_Vector3i_plusZ();

    TEST_ASSERT_INT_EQUAL(vec.x, 0);
    TEST_ASSERT_INT_EQUAL(vec.y, 0);
    TEST_ASSERT_INT_EQUAL(vec.z, 1);
}
