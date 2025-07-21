#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshSubdivide.h>
#include <MRCMesh/MRTorus.h>

void testMeshSubdivide(void)
{
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 16;
    int32_t secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    MR_SubdivideSettings* subdivideSettings = MR_SubdivideSettings_DefaultConstruct();
    MR_SubdivideSettings_Set_maxDeviationAfterFlip( subdivideSettings, 0.5f );
    MR_subdivideMesh_MR_Mesh(mesh, subdivideSettings);

    MR_SubdivideSettings_Destroy(subdivideSettings);
    MR_Mesh_Destroy(mesh);
}
