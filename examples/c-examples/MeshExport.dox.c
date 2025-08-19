#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRString.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRVector3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );

    MR_Mesh* mesh = MR_makeCube( &size, &base );

    // extract vertices
    const MR_Vector3f* vertices = MR_VertCoords_data_const( MR_Mesh_Get_points( mesh ) );
    size_t verticesNum = MR_VertCoords_size( MR_Mesh_Get_points( mesh ) );
    // you can access coordinates via struct fields...
    printf( "Vertex coordinates:\n" );
    for ( size_t i = 0; i < verticesNum; ++i )
        printf( "  vertex %d:  % f; % f; % f\n", (int)i, vertices[i].x, vertices[i].y, vertices[i].z );
    // ...or cast them to a row-major float array
    float* vertexData = malloc( sizeof( float ) * 3 * verticesNum );
    memcpy( vertexData, vertices, sizeof( MR_Vector3f ) * verticesNum );

    // extract faces
    MR_Triangulation* t = MR_MeshTopology_getTriangulation( MR_Mesh_Get_topology( mesh ) );
    const MR_std_array_MR_VertId_3 *faces = MR_Triangulation_data_const( t );
    size_t facesNum = MR_Triangulation_size( t );
    // faces are stored as vertex id triples...
    printf( "Face vertex IDs:\n" );
    for ( size_t i = 0; i < verticesNum; ++i )
        printf( "  face %d:  %d, %d, %d\n", (int)i, faces[i].elems[0].id_, faces[i].elems[1].id_, faces[i].elems[2].id_ );
    // ...and can also be cast to an integer array
    int* faceData = malloc( sizeof( int ) * 3 * facesNum );
    memcpy( faceData, faces, sizeof( MR_std_array_MR_VertId_3 ) * facesNum );

    MR_Triangulation_Destroy( t );
    MR_Mesh_Destroy( mesh );
    free( vertexData );
    free( faceData );
    return EXIT_SUCCESS;
}
