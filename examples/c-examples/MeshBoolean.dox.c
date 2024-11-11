#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshBoolean.h"
#include "MRMeshC/MRMeshLoad.h"
#include "MRMeshC/MRMeshSave.h"
#include "MRMeshC/MRString.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if ( argc != 5 )
    {
        fprintf( stderr, "Usage: %s { unite | intersect } INPUT1 INPUT2 OUTPUT", argv[0] );
        goto out;
    }

    MRBooleanOperation op;
    if ( strcmp( argv[1], "unite" ) == 0 )
        op = MRBooleanOperationUnion;
    else if ( strcmp( argv[1], "intersect" ) == 0 )
        op = MRBooleanOperationIntersection;
    else
    {
        fprintf( stderr, "Incorrect operation: %s", argv[1] );
        goto out;
    }

    const char* input1 = argv[2];
    const char* input2 = argv[3];
    const char* output = argv[4];

    // error messages will be stored here
    MRString* errorString = NULL;

    MRMesh* mesh1 = mrMeshLoadFromAnySupportedFormat( input1, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load mesh 1: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    MRMesh* mesh2 = mrMeshLoadFromAnySupportedFormat( input2, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load mesh 2: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_mesh1;
    }

    // you can set some parameters for boolean, e.g. progress callback
    MRBooleanParameters params = mrBooleanParametersNew();
    params.cb = onProgress;
    // perform the boolean operation
    MRBooleanResult result = mrBoolean( mesh1, mesh2, op, &params );
    if ( result.errorString )
    {
        fprintf( stderr, "Failed to perform boolean: %s", mrStringData( result.errorString ) );
        mrStringFree( errorString );
        goto out_mesh2;
    }

    mrMeshSaveToAnySupportedFormat( result.mesh, output, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save result: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_result;
    }

    rc = EXIT_SUCCESS;
out_result:
    mrMeshFree( result.mesh );
out_mesh2:
    mrMeshFree( mesh2 );
out_mesh1:
    mrMeshFree( mesh1 );
out:
    return rc;
}
