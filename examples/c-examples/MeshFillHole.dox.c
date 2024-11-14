#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshFillHole.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>

#define MIN_HOLE_AREA 100.f

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

    // get list of existing holes; each hole is represented by a single edge lying on the hole's border
    MREdgePath* holes = mrMeshFindHoleRepresentiveEdges( mesh );
    if ( holes->size == 0 )
    {
        printf( "Mesh doesn't have any holes" );
        goto out_holes;
    }

    // you can set various parameters for the fill hole process; see the documentation for more info
    MRFillHoleParams params = mrFillHoleParamsNew();
    // think of a metric as a method to fill holes in a preferred way
    // you can define one or choose from one of predefined metrics from MRMeshMetrics.h
    MRFillHoleMetric* metric = mrGetUniversalMetric( mesh );
    params.metric = metric;
    // optionally get the bitset of created faces
    MRFaceBitSet* newFaces = mrFaceBitSetNew( 0, false );
    params.outNewFaces = newFaces;

    // you can either fill all holes at once or one by one
    // in the second case don't forget to check the output fields of params (e.g. outNewFaces) after every iteration
    size_t newFaceCount = 0;
#define FILL_ALL_HOLES 1
#if FILL_ALL_HOLES
    mrFillHoles( mesh, holes->data, holes->size, &params );
    newFaceCount = mrBitSetCount( newFaces );
#else
    for ( int i = 0; i < mrEdgePathSize( holes ); i++ )
    {
        MREdgeId e = mrEdgePathData( holes )[i];
        MRVector3f holeDirArea = mrMeshHoleDirArea( mesh, e );
        if ( mrVector3Length( &holeDirArea ) >= MIN_HOLE_AREA )
        {
            mrFillHole( mesh, e, &params );
            newFaceCount += mrBitSetCount( newFaces );
        }
    }
#endif

    printf( "Added new %zu faces", newFaceCount );

    mrMeshSaveToAnySupportedFormat( mesh, output, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_newFaces;
    }

    rc = EXIT_SUCCESS;
out_newFaces:
    mrFaceBitSetFree( newFaces );
out_metric:
    mrFillHoleMetricFree( metric );
out_holes:
    mrEdgePathFree( holes );
out_mesh:
    mrMeshFree( mesh );
out:
    return rc;
}
