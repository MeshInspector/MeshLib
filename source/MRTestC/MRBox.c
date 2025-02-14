#include "TestMacros.h"

#include "MRMesh.h"
#include "MRMeshC/MRbox.h"

void testBox(void)
{
	MRVector3f min = mrVector3fDiagonal(-1.f);
	MRVector3f size = mrVector3fDiagonal(1.f);
	MRBox3f box = mrBox3fFromMinAndSize(&min, &size);
	float volume = mrBox3fVolume(&box);
	TEST_ASSERT(volume > 0.999f && volume < 1.001f);
}