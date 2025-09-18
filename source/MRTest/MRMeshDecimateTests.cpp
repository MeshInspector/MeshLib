#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBuffer.h>

namespace MR
{

// check if Decimator updates region
TEST( MRMesh, MeshDecimate )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettings decimateSettings;
    decimateSettings.maxError = 0.001f;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    auto decimateResults = decimateMesh(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);
}

TEST( MRMesh, MeshDecimateParallel )
{
    const int cNumVerts = 400;
    auto mesh = makeSphere( { .numMeshVertices = cNumVerts } );
    mesh.packOptimally();
    DecimateSettings settings
    {
        .maxError = 1000000, // no actual limit
        .maxDeletedVertices = cNumVerts - 3,
        .subdivideParts = 8
    };
    decimateMesh( mesh, settings );
    ASSERT_EQ( mesh.topology.numValidFaces(), 2 );
    ASSERT_EQ( mesh.topology.numValidVerts(), 3 );
}

} //namespace MR
