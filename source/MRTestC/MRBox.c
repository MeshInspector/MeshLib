#include "TestMacros.h"

#include "MRMeshC/MRBox.h"

void testBoxf(void)
{
	MRVector3f min = mrVector3fDiagonal(-1.f);
	MRVector3f size = mrVector3fDiagonal(1.f);

	MRBox3f box = mrBox3fFromMinAndSize(&min, &size);

	float volume = mrBox3fVolume(&box);
	TEST_ASSERT(volume > 0.999f && volume < 1.001f);

	float diag = mrBox3fDiagonal(&box);

	TEST_ASSERT(diag > 1.73f && diag < 1.74f);

	TEST_ASSERT(mrBox3fValid(&box) == true);

	MRVector3f boxSize = mrBox3fSize(&box);

	TEST_ASSERT(boxSize.x > 0.999f && boxSize.x < 1.001f);
	TEST_ASSERT(boxSize.y > 0.999f && boxSize.y < 1.001f);
	TEST_ASSERT(boxSize.z > 0.999f && boxSize.z < 1.001f);

	MRVector3f boxCenter = mrBox3fCenter(&box);

	TEST_ASSERT(boxCenter.x < -0.499f && boxCenter.x > -0.501f);
	TEST_ASSERT(boxCenter.y < -0.499f && boxCenter.y > -0.501f);
	TEST_ASSERT(boxCenter.z < -0.499f && boxCenter.z > -0.501f);
}

void testBoxfInvalid(void)
{
	MRBox3f invBox = mrBox3fNew();
	TEST_ASSERT(mrBox3fValid(&invBox) == false);
}

void testBoxi(void)
{
	MRVector3i min = mrVector3iDiagonal(-2);
	MRVector3i size = mrVector3iDiagonal(2);

	MRBox3i box = mrBox3iFromMinAndSize(&min, &size);

	int volume = mrBox3iVolume(&box);
	TEST_ASSERT( volume == 8);

	TEST_ASSERT(mrBox3iValid(&box) == true);

	MRVector3i boxSize = mrBox3iSize(&box);

	TEST_ASSERT(boxSize.x == 2);
	TEST_ASSERT(boxSize.y == 2);
	TEST_ASSERT(boxSize.z == 2);

	MRVector3i boxCenter = mrBox3iCenter(&box);

	TEST_ASSERT(boxCenter.x == -1);
	TEST_ASSERT(boxCenter.y == -1);
	TEST_ASSERT(boxCenter.z == -1);
}

void testBoxiInvalid(void)
{
	MRBox3i invBox = mrBox3iNew();
	TEST_ASSERT(mrBox3iValid(&invBox) == false);
}
