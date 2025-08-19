#include "TestMacros.h"

#include "MRMeshC/MRAddNoise.h"
#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRTorus.h>

void testAddNoise(void)
{
	MRMakeTorusParameters paramsTorus = { 1.1f, 0.5f, 16, 16 };
	MRMesh* mesh = mrMakeTorus(&paramsTorus);

	MRNoiseSettings noiseSettings = mrNoiseSettingsNew();
	mrAddNoiseToMesh(mesh, NULL, &noiseSettings, NULL);

	mrMeshFree(mesh);
}
