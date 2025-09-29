#include "TestMacros.h"

#include "MRCMesh/MRAddNoise.h"
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRTorus.h>

void testAddNoise(void)
{
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int primaryResolution = 16;
    int secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    MR_addNoise_MR_Mesh(mesh, NULL, NULL);

    MR_Mesh_Destroy(mesh);
}
