#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshSubdivide.h>
#include <MRCMesh/MRTorus.h>

void testMeshSubdivide(void)
{
    MRMakeTorusParameters paramsTorus = { 1.1f, 0.5f, 16, 16 };
    MRMesh* mesh = mrMakeTorus(&paramsTorus);

    MRSubdivideSettings subdivideSettings = mrSubdivideSettingsNew();
    subdivideSettings.maxDeviationAfterFlip = 0.5f;
    mrSubdivideMesh(mesh, &subdivideSettings);

    mrMeshFree(mesh);
}
