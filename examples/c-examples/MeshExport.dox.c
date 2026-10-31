#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/std_vector_std_array_MR_VertId_3.h>

#include <stdlib.h>

int main( void )
{
    // create some mesh
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );
    MR_Mesh* mesh = MR_makeCube( &size, &base );

    // all vertices of valid triangles
    MR_std_vector_std_array_MR_VertId_3* triangles = MR_MeshTopology_getAllTriVerts( MR_Mesh_Get_topology( mesh ) );

    // all point coordinates as x,y,z triples
    const float* points = (const float*)MR_VertCoords_data( MR_Mesh_Get_points( mesh ) );
    // triangle vertices as triples of ints (pointing to elements in points vector)
    const int* vertexTriples = (const int*)MR_std_vector_std_array_MR_VertId_3_data( triangles );

    // TODO: export in your format
    (void)points;
    (void)vertexTriples;

    MR_std_vector_std_array_MR_VertId_3_Destroy( triangles );
    MR_Mesh_Destroy( mesh );
    return EXIT_SUCCESS;
}
