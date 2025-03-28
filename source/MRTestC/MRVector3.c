#include "TestMacros.h"
#include "MRMeshC/MRVector3.h"

void testMrVector3fDiagonal(void) {
    MRVector3f vec = mrVector3fDiagonal(3.0f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 3.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 3.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 3.0f, 0.0001f);
}

void testMrVector3fPlusX(void) {
    MRVector3f vec = mrVector3fPlusX();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 0.0f, 0.0001f);
}

void testMrVector3fPlusY(void) {
    MRVector3f vec = mrVector3fPlusY();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 0.0f, 0.0001f);
}

void testMrVector3fPlusZ(void) {
    MRVector3f vec = mrVector3fPlusZ();

    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.x, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.y, 0.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(vec.z, 1.0f, 0.0001f);
}

void testMrVector3fAdd(void) {
    MRVector3f a = {1.0f, 2.0f, 3.0f};
    MRVector3f b = {4.0f, 5.0f, 6.0f};
    MRVector3f result = mrVector3fAdd(&a, &b);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 5.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 7.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 9.0f, 0.0001f);
}

void testMrVector3fSub(void) {
    MRVector3f a = {5.0f, 7.0f, 9.0f};
    MRVector3f b = {4.0f, 5.0f, 6.0f};
    MRVector3f result = mrVector3fSub(&a, &b);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 1.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 3.0f, 0.0001f);
}

void testMrVector3fMulScalar(void) {
    MRVector3f vec = {1.0f, 2.0f, 3.0f};
    float scalar = 2.0f;
    MRVector3f result = mrVector3fMulScalar(&vec, scalar);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 2.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 4.0f, 0.0001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 6.0f, 0.0001f);
}

void testMrVector3fLengthSq(void) {
    MRVector3f vec = {1.0f, 2.0f, 3.0f};
    float lengthSq = mrVector3fLengthSq(&vec);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(lengthSq, 14.0f, 0.0001f);
}

void testMrVector3fLength(void) {
    MRVector3f vec = {3.0f, 4.0f, 0.0f};
    float length = mrVector3fLength(&vec);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(length, 5.0f, 0.0001f);
}

void testMrVector3iDiagonal(void) {
    MRVector3i vec = mrVector3iDiagonal(3);

    TEST_ASSERT_INT_EQUAL(vec.x, 3);
    TEST_ASSERT_INT_EQUAL(vec.y, 3);
    TEST_ASSERT_INT_EQUAL(vec.z, 3);
}

void testMrVector3iPlusX(void) {
    MRVector3i vec = mrVector3iPlusX();

    TEST_ASSERT_INT_EQUAL(vec.x, 1);
    TEST_ASSERT_INT_EQUAL(vec.y, 0);
    TEST_ASSERT_INT_EQUAL(vec.z, 0);
}

void testMrVector3iPlusY(void) {
    MRVector3i vec = mrVector3iPlusY();

    TEST_ASSERT_INT_EQUAL(vec.x, 0);
    TEST_ASSERT_INT_EQUAL(vec.y, 1);
    TEST_ASSERT_INT_EQUAL(vec.z, 0);
}

void testMrVector3iPlusZ(void) {
    MRVector3i vec = mrVector3iPlusZ();

    TEST_ASSERT_INT_EQUAL(vec.x, 0);
    TEST_ASSERT_INT_EQUAL(vec.y, 0);
    TEST_ASSERT_INT_EQUAL(vec.z, 1);
}
