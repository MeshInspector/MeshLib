#include "TestMacros.h"

#include "MRCMesh/MRBox.h"

void testBoxf(void)
{
    MR_Vector3f min = MR_Vector3f_diagonal(-1.f);
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);

    MR_Box3f box = MR_Box3f_fromMinAndSize(&min, &size);

    float volume = MR_Box3f_volume(&box);
    TEST_ASSERT(volume > 0.999f && volume < 1.001f);

    float diag = MR_Box3f_diagonal(&box);

    TEST_ASSERT(diag > 1.73f && diag < 1.74f);

    TEST_ASSERT(MR_Box3f_valid(&box) == true);

    MR_Vector3f boxSize = MR_Box3f_size(&box);

    TEST_ASSERT(boxSize.x > 0.999f && boxSize.x < 1.001f);
    TEST_ASSERT(boxSize.y > 0.999f && boxSize.y < 1.001f);
    TEST_ASSERT(boxSize.z > 0.999f && boxSize.z < 1.001f);

    MR_Vector3f boxCenter = MR_Box3f_center(&box);

    TEST_ASSERT(boxCenter.x < -0.499f && boxCenter.x > -0.501f);
    TEST_ASSERT(boxCenter.y < -0.499f && boxCenter.y > -0.501f);
    TEST_ASSERT(boxCenter.z < -0.499f && boxCenter.z > -0.501f);
}

void testBoxfInvalid(void)
{
    MR_Box3f invBox = MR_Box3f_DefaultConstruct();
    TEST_ASSERT(MR_Box3f_valid(&invBox) == false);
}

void testBoxi(void)
{
    MR_Vector3i min = MR_Vector3i_diagonal(-2);
    MR_Vector3i size = MR_Vector3i_diagonal(2);

    MR_Box3i box = MR_Box3i_fromMinAndSize(&min, &size);

    int volume = MR_Box3i_volume(&box);
    TEST_ASSERT( volume == 8);

    TEST_ASSERT(MR_Box3i_valid(&box) == true);

    MR_Vector3i boxSize = MR_Box3i_size(&box);

    TEST_ASSERT(boxSize.x == 2);
    TEST_ASSERT(boxSize.y == 2);
    TEST_ASSERT(boxSize.z == 2);

    MR_Vector3i boxCenter = MR_Box3i_center(&box);

    TEST_ASSERT(boxCenter.x == -1);
    TEST_ASSERT(boxCenter.y == -1);
    TEST_ASSERT(boxCenter.z == -1);
}

void testBoxiInvalid(void)
{
    MR_Box3i invBox = MR_Box3i_DefaultConstruct();
    TEST_ASSERT(MR_Box3i_valid(&invBox) == false);
}
