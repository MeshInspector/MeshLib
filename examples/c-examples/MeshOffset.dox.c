#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshLoad.h"
#include "MRMeshC/MRMeshSave.h"
#include "MRMeshC/MROffset.h"
#include "MRMeshC/MRString.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define APPROX_VOXEL_COUNT 10000000.f

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
    if ( argc != 3 && argc != 4 )
    {
        fprintf( stderr, "Usage: %s OFFSET_VALUE INPUT [OUTPUT]", argv[0] );
        goto out;
    }

    float offsetValue = atof( argv[1] );
    if ( !isfinite( offsetValue ) )
    {
        fprintf( stderr, "Incorrect offset value: %s", argv[1] );
        goto out;
    }

    const char* input = argv[2];
    const char* output = ( argc == 3 ) ? argv[2] : argv[3];

    // error messages will be stored here
    MRString* errorString = NULL;

    MRMesh* inputMesh = mrMeshLoadFromAnySupportedFormat( input, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load inputMesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // offset functions can also be applied to separate mesh components rather than to the whole mesh
    // this is not our case, so the region is set to NULL
    MRMeshPart inputMeshPart = (MRMeshPart){
        .mesh = inputMesh,
        .region = NULL,
    };

    MROffsetParameters params = mrOffsetParametersNew();
    // calculate voxel size depending on desired accuracy and/or memory consumption
    params.voxelSize = mrSuggestVoxelSize( inputMeshPart, APPROX_VOXEL_COUNT );
    // set optional progress callback
    params.callBack = onProgress;

    MRMesh* outputMesh = mrOffsetMesh( inputMeshPart, offsetValue, &params, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to perform offset: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_inputMesh;
    }

    mrMeshSaveToAnySupportedFormat( outputMesh, output, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save inputMesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_outputMesh;
    }

    rc = EXIT_SUCCESS;
out_outputMesh:
    mrMeshFree( outputMesh );
out_inputMesh:
    mrMeshFree( inputMesh );
out:
    return rc;
}
