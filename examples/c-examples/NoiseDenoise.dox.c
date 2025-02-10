#include <MRMeshC/MRAddNoise.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRNormalDenoising.h>

#include <stdlib.h>

int main( int argc, char* argv[] )
{
    // Load mesh
    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( "mesh.stl", NULL );

    // Add noise to the mesh
    MRBox3f box = mrMeshComputeBoundingBox( mesh, NULL );
    MRNoiseSettings noiseSettings = mrNoiseSettingsNew();
    noiseSettings.sigma = mrBox3fDiagonal( &box ) * 0.0001f;
    mrAddNoiseToMesh( mesh, NULL, &noiseSettings, NULL );

    // Invalidate the mesh because of the external vertex changes
    mrMeshInvalidateCaches( mesh, true );

    // Save the noised mesh
    mrMeshSaveToAnySupportedFormat( mesh, "noised_mesh.stl", NULL, NULL );

    // Denoise the mesh with sharpening for sharp edges
    // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    mrMeshDenoiseViaNormals( mesh, NULL, NULL );

    // Save the denoised mesh
    mrMeshSaveToAnySupportedFormat( mesh, "denoised_mesh.stl", NULL, NULL );

    mrMeshFree( mesh );
    return EXIT_SUCCESS;
}
