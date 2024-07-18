#include "MRMeshC/MRMultiwayICP.h"
#include "MRMeshC/MRPointCloud.h"
#include "MRMeshC/MRPointsLoad.h"
#include "MRMeshC/MRPointsSave.h"
#include "MRMeshC/MRString.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// print progress every 10%
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
    if ( argc < 4 )
    {
        fprintf( stderr, "Usage: %s INPUT1 INPUT2 [INPUTS...] OUTPUT", argv[0] );
        goto out;
    }

    // error messages will be stored here
    MRString* errorString = NULL;

    const int inputNum = argc - 2;
    MRPointCloud** inputs = malloc( sizeof( MRPointCloud* ) * inputNum );
    memset( inputs, 0, sizeof( MRPointCloud* ) * inputNum );
    MRMeshOrPointsXf** inputXfs = malloc( sizeof( MRMeshOrPointsXf* ) * inputNum );
    memset( inputXfs, 0, sizeof( MRMeshOrPointsXf* ) * inputNum );
    for ( int i = 0; i < inputNum; ++i )
    {
        inputs[i] = mrPointsLoadFromAnySupportedFormat( argv[1 + i], &errorString );
        if ( errorString )
        {
            fprintf( stderr, "Failed to load point cloud: %s", mrStringData( errorString ) );
            mrStringFree( errorString );
            goto out_inputs;
        }
        inputXfs[i] = mrMeshOrPointsXfFromPointCloud( inputs[i], NULL );
    }

    MRICPProperties params = mrICPPropertiesNew();

    MRMultiwayICPSamplingParameters samplingParams = mrMultiwayIcpSamplingParametersNew();
    samplingParams.cb = onProgress;

    MRMultiwayICP* icp = mrMultiwayICPNew( *inputXfs, inputNum, &samplingParams );
    mrMultiwayICPSetParams( icp, &params );
    mrMultiwayICPUpdateAllPointPairs( icp, NULL );
    MRVectorAffineXf3f* xfs = mrMultiwayICPCalculateTransformations( icp, NULL );

    MRPointCloud* output = mrPointCloudNew();
    for ( int i = 0; i < inputNum; i++ )
    {
        const MRAffineXf3f* xf = mrVectorAffineXf3fData( xfs ) + i;
        for ( int j = 0; j < mrPointCloudPointsNum( inputs[i] ); j++ )
        {
            MRVector3f* point = mrPointCloudPointsRef( inputs[i] ) + j;
            mrAffineXf3fApply( xf, point );
            mrPointCloudAddPoint( output, point );
        }
    }

    mrPointsSaveToAnySupportedFormat( output, argv[argc - 1], &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save point cloud: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_output;
    }

    rc = EXIT_SUCCESS;
out_output:
    mrPointCloudFree( output );
out_xfs:
    mrVectorAffineXf3fFree( xfs );
out_icp:
    mrMultiwayICPFree( icp );
out_inputs:
    for ( int i = 0; i < inputNum; i++ )
    {
        mrMeshOrPointsXfFree( inputXfs[i] );
        mrPointCloudFree( inputs[i] );
    }
    free( inputXfs );
    free( inputs );
out:
    return rc;
}
