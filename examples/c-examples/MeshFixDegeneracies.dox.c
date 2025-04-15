#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshFixer.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    if ( argc != 2 && argc != 3 )
    {
        fprintf( stderr, "Usage: %s INPUT [OUTPUT]", argv[0] );
        goto out;
    }

    const char* input = argv[1];
    const char* output = ( argc == 2 ) ? argv[1] : argv[2];

    // error messages will be stored here
    MRString* errorString = NULL;

    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( input, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // you can set various parameters for the resolving process; see the documentation for more info
    MRFixMeshDegeneraciesParams params = mrFixMeshDegeneraciesParamsNew();
    // maximum permitted deviation
    const MRBox3f bbox = mrMeshComputeBoundingBox( mesh, NULL );
    params.maxDeviation = 1e-5f * mrBox3fDiagonal( &bbox );
    // maximum length of edges to be collapsed
    params.tinyEdgeLength = 1e-3f;

    mrFixMeshDegeneracies( mesh, &params, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to fix mesh degeneracies: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_mesh;
    }

    MRSaveSettings saveSettings = mrSaveSettingsNew();
    mrMeshSaveToAnySupportedFormat( mesh, output, &saveSettings, &errorString);
    if ( errorString )
    {
        fprintf( stderr, "Failed to save mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_mesh;
    }

    rc = EXIT_SUCCESS;
out_mesh:
    mrMeshFree( mesh );
out:
    return rc;
}
