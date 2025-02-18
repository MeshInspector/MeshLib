#include <MRMeshC/MRFreeFormDeformer.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>

#include <stdlib.h>

int main( int argc, char* argv[] )
{
    // Load mesh
    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( "mesh.stl", NULL );

    // Compute mesh bounding box
    MRBox3f box = mrMeshComputeBoundingBox( mesh, NULL );

    // Construct deformer on mesh vertices
    MRFreeFormDeformer* ffDeformer = mrFreeFormDeformerNewFromMesh( mesh, NULL );

    // Init deformer with 3x3 grid on mesh box
    MRVector3i resolution = mrVector3iDiagonal( 3 );
    mrFreeFormDeformerInit( ffDeformer, &resolution, &box );

    // Move some control points of the grid to the center
    MRVector3i controlPoints[] = {
        { 1, 1, 0 },
        { 1, 1, 2 },
        { 0, 1, 1 },
        { 2, 1, 1 },
        { 1, 0, 1 },
        { 1, 2, 1 },
    };
    MRVector3f center = mrBox3fCenter( &box );
    for ( int i = 0; i < 6; ++i )
        mrFreeFormDeformerSetRefGridPointPosition( ffDeformer, &controlPoints[i], &center );

    // Apply the deformation to the mesh vertices
    mrFreeFormDeformerApply( ffDeformer );

    // Invalidate the mesh because of external vertex changes
    mrMeshInvalidateCaches( mesh, true );

    // Save deformed mesh
    mrMeshSaveToAnySupportedFormat( mesh, "deformed_mesh.stl", NULL, NULL );

    mrFreeFormDeformerFree( ffDeformer );
    mrMeshFree( mesh );
    return EXIT_SUCCESS;
}
