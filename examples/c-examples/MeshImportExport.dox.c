#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshLoad.h"
#include "MRMeshC/MRMeshSave.h"
#include "MRMeshC/MRMeshTopology.h"
#include "MRMeshC/MRString.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    if ( argc != 3 )
    {
        fprintf( stderr, "Usage: %s INPUTFILE OUTPUTFILE", argv[0] );
        goto out;
    }
    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    // error messages will be stored here
    MRString* errorString = NULL;

    MRMesh* mesh1 = mrMeshLoadFromAnySupportedFormat( inputFile, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // extract vertices
    const MRVector3f* vertices = mrMeshPoints( mesh1 );
    size_t verticesNum = mrMeshPointsNum( mesh1 );
    // you can access coordinates via struct fields...
    if ( verticesNum > 0 )
        printf( "The first vertex coordinates: %f; %f; %f", vertices[0].x, vertices[0].y, vertices[0].z );
    // ..or cast them to a row-major float array
    float* vertexData = malloc( sizeof( float ) * 3 * verticesNum );
    memcpy( vertexData, vertices, sizeof( MRVector3f ) * verticesNum );

    // extract faces
    MRTriangulation* t = mrMeshGetTriangulation( mesh1 );
    const MRThreeVertIds* faces = t->data;
    size_t facesNum = t->size;
    // faces are stored as vertex id triples...
    if ( facesNum > 0 )
        printf( "The first face's vertex ids: %d, %d, %d", faces[0][0].id, faces[0][1].id, faces[0][2].id );
    // ...and can also be casted to an integer array
    int* faceData = malloc( sizeof( int ) * 3 * facesNum );
    memcpy( faceData, faces, sizeof( MRThreeVertIds ) * facesNum );

    // meshes can be constructed from these data arrays
    MRMesh* mesh2 = mrMeshFromTriangles( (const MRVector3f*)vertexData, verticesNum, (const MRThreeVertIds*)faceData, facesNum );

    mrMeshSaveToAnySupportedFormat( mesh2, outputFile, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_mesh2;
    }

    rc = EXIT_SUCCESS;
out_mesh2:
    mrMeshFree( mesh2 );
out_faceData:
    free( faceData );
out_t:
    mrTriangulationFree( t );
out_vertexData:
    free( vertexData );
out_mesh1:
    mrMeshFree( mesh1 );
out:
    return rc;
}
