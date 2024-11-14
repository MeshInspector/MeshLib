#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshDecimate.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>

// print progress every 10%
int gProgress = -1;
bool onProgress( float v )
{
    int progress = (int)( 10.f * v );
    if ( progress != gProgress )
    {
        gProgress = progress;
        printf( "%d%%...\n", progress * 10 );
    }
    return true;
}

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

    // you can set various parameters for the mesh decimation; see the documentation for more info
    MRDecimateSettings params = mrDecimateSettingsNew();
    params.strategy = MRDecimateStrategyMinimizeError;
    // maximum permitted deviation
    const MRBox3f bbox = mrMeshComputeBoundingBox( mesh, NULL );
    params.maxError = 1e-5f * mrBox3fDiagonal( &bbox );
    // maximum length of edges to be collapsed
    params.tinyEdgeLength = 1e-3f;
    // pack mesh after decimation
    params.packMesh = true;
    // set progress callback
    params.progressCallback = onProgress;

    MRDecimateResult result = mrDecimateMesh( mesh, &params );
    if ( !result.cancelled )
        printf( "Removed %d vertices, %d faces", result.vertsDeleted, result.facesDeleted );
    else
    {
        fprintf( stderr, "Cancelled" );
        goto out_mesh;
    }

    mrMeshSaveToAnySupportedFormat( mesh, output, &errorString );
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
