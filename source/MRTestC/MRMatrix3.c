#include "TestMacros.h"
#include "MRMeshC/MRMatrix3.h"


void testMrMatrix3fIdentity() {
    MRMatrix3f identityMatrix = mrMatrix3fIdentity();

    // Check that the diagonal values are 1 and off-diagonal are 0
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.x.x, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.y.y, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.z.z, 1.0f, 0.001f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.x.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.x.z, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.y.x, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.y.z, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.z.x, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(identityMatrix.z.y, 0.0f, 0.001f);
}

void testMrMatrix3fRotationScalar() {
    MRVector3f axis = {1.0f, 0.0f, 0.0f}; // Rotation around X-axis
    float angle = 3.14159f / 2.0f; // 90 degrees

    MRMatrix3f rotationMatrix = mrMatrix3fRotationScalar(&axis, angle);

    // Expected results for a 90-degree rotation around X-axis
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.x.x, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.y.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.y.z, -1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.z.y, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.z.z, 0.0f, 0.001f);
}

void testMrMatrix3fRotationVector() {
    MRVector3f from = {1.0f, 0.0f, 0.0f}; // Initial vector (x-axis)
    MRVector3f to = {0.0f, 1.0f, 0.0f}; // Target vector (y-axis)

    MRMatrix3f rotationMatrix = mrMatrix3fRotationVector(&from, &to);

    // Expected results for a 90-degree rotation from x-axis to y-axis
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.x.x, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.x.y, -1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.x.z, 0.0f, 0.001f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.y.x, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.y.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.y.z, 0.0f, 0.001f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.z.x, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.z.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(rotationMatrix.z.z, 1.0f, 0.001f);
}

void testMrMatrix3fAdd() {
    MRMatrix3f matrixA = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    MRMatrix3f matrixB = {
        {9.0f, 8.0f, 7.0f},
        {6.0f, 5.0f, 4.0f},
        {3.0f, 2.0f, 1.0f}
    };

    MRMatrix3f result = mrMatrix3fAdd(&matrixA, &matrixB);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.x, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.y, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.z, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.x, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.y, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.z, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.x, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.y, 10.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.z, 10.0f, 0.001f);
}

void testMrMatrix3fSub() {
    MRMatrix3f matrixA = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    MRMatrix3f matrixB = {
        {9.0f, 8.0f, 7.0f},
        {6.0f, 5.0f, 4.0f},
        {3.0f, 2.0f, 1.0f}
    };

    MRMatrix3f result = mrMatrix3fSub(&matrixA, &matrixB);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.x, -8.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.y, -6.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.z, -4.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.x, -2.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.z, 2.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.x, 4.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.y, 6.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.z, 8.0f, 0.001f);
}


void testMrMatrix3fMul() {
    MRMatrix3f matrixA = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    MRMatrix3f matrixB = {
        {2.0f, 1.0f, 0.0f},
        {1.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 3.0f}
    };

    MRMatrix3f result = mrMatrix3fMul(&matrixA, &matrixB);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.x, 2.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.y, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x.z, 0.0f, 0.001f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.x, 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.y, 2.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y.z, 0.0f, 0.001f);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.x, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.y, 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z.z, 3.0f, 0.001f);
}

void testMrMatrix3fMulVector() {
    MRMatrix3f matrix = {
        {2.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 2.0f}
    };

    MRVector3f vector = {1.0f, 2.0f, 3.0f};

    MRVector3f result = mrMatrix3fMulVector(&matrix, &vector);

    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.x, 2.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.y, 4.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQUAL_APPROX(result.z, 6.0f, 0.001f);
}

void testMrMatrix3fEqual() {
    MRMatrix3f matrixA = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    MRMatrix3f matrixB = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    MRMatrix3f matrixC = {
        {2.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 2.0f}
    };

    TEST_ASSERT(mrMatrix3fEqual(&matrixA, &matrixB)); // Should be true
    TEST_ASSERT(!mrMatrix3fEqual(&matrixA, &matrixC)); // Should be false
}
