#include "TestMacros.h"

#include "MRCMesh/MRAddNoise.h"
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRTorus.h>

void testAddNoise(void)
{
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 16;
    int32_t secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    MR_addNoise_MR_Mesh(mesh, NULL, NULL);

    MR_Mesh_Destroy(mesh);
}
