#include <MRMeshC/MRBox.h>
#include <MRMeshC/MRMultiwayICP.h>
#include <MRMeshC/MRPointCloud.h>
#include <MRMeshC/MRPointsLoad.h>
#include <MRMeshC/MRPointsSave.h>
#include <MRMeshC/MRString.h>

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

    // the global registration can be applied to meshes and point clouds
    // to simplify the sample app, we will work with point clouds only
    const int inputNum = argc - 2;
    MRPointCloud** inputs = malloc( sizeof( MRPointCloud* ) * inputNum );
    memset( inputs, 0, sizeof( MRPointCloud* ) * inputNum );
    MRMeshOrPointsXf** inputXfs = malloc( sizeof( MRMeshOrPointsXf* ) * inputNum );
    memset( inputXfs, 0, sizeof( MRMeshOrPointsXf* ) * inputNum );
    MRBox3f maxBBox = mrBox3fNew();
    for ( int i = 0; i < inputNum; ++i )
    {
        inputs[i] = mrPointsLoadFromAnySupportedFormat( argv[1 + i], &errorString );
        if ( errorString )
        {
            fprintf( stderr, "Failed to load point cloud: %s", mrStringData( errorString ) );
            mrStringFree( errorString );
            goto out_inputs;
        }

        inputXfs[i] = mrMeshOrPointsXfFromPointCloud( inputs[i], NULL ); // or mrMeshOrPointsXfFromMesh for meshes

        MRBox3f bbox = mrPointCloudComputeBoundingBox( inputs[i], NULL );
        if ( !mrBox3fValid( &maxBBox ) || mrBox3fVolume( &bbox ) > mrBox3fVolume( &maxBBox ) )
            maxBBox = bbox;
    }

    MRMultiwayICPSamplingParameters samplingParams = mrMultiwayIcpSamplingParametersNew();
    samplingParams.samplingVoxelSize = mrBox3fDiagonal( &maxBBox ) * 0.03f;
    samplingParams.cb = onProgress;

    MRMultiwayICP* icp = mrMultiwayICPNew( *inputXfs, inputNum, &samplingParams );

    MRICPProperties params = mrICPPropertiesNew();
    mrMultiwayICPSetParams( icp, &params );

    // gather statistics
    mrMultiwayICPUpdateAllPointPairs( icp, NULL );
    size_t numSamples = mrMultiWayICPGetNumSamples( icp );
    size_t numActivePairs = mrMultiWayICPGetNumActivePairs( icp );
    printf( "Samples: %zu", numSamples );
    printf( "Active point pairs: %zu", numActivePairs );
    if ( numActivePairs > 0 )
    {
        double p2ptMetric = mrMultiWayICPGetMeanSqDistToPoint( icp, NULL );
        double p2ptInaccuracy = mrMultiWayICPGetMeanSqDistToPoint( icp, &p2ptMetric );
        printf( "RMS point-to-point distance: %f ± %f", p2ptMetric, p2ptInaccuracy );

        double p2plMetric = mrMultiWayICPGetMeanSqDistToPlane( icp, NULL );
        double p2plInaccuracy = mrMultiWayICPGetMeanSqDistToPlane( icp, &p2plMetric );
        printf( "RMS point-to-plane distance: %f ± %f", p2plMetric, p2plInaccuracy );
    }

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
