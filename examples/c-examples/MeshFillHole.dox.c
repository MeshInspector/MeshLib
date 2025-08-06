#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshFillHole.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshMetrics.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRString.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/std_vector_MR_EdgeId.h>

#include <stdio.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    if ( argc != 2 && argc != 3 )
    {
        fprintf( stderr, "Usage: %s INPUT [OUTPUT]", argv[0] );
        return EXIT_FAILURE;
    }

    const char* input = argv[1];
    const char* output = ( argc == 2 ) ? argv[1] : argv[2];

    int rc = EXIT_FAILURE;


    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( input, NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    // Handle failure to load mesh.
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail_mesh_loading;
    }


    // Get the list of the existing holes; each hole is represented by a single edge from the hole's border.
    MR_std_vector_MR_EdgeId* holes = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    if ( MR_std_vector_MR_EdgeId_IsEmpty( holes ) )
    {
        printf( "Mesh doesn't have any holes" );
        goto fail_no_holes;
    }

    // You can set various parameters for the hole filling process; see the documentation for more info.
    MR_FillHoleParams* params = MR_FillHoleParams_DefaultConstruct();
    // The metric controls how exactly the hole is filled.
    // You can make a custom one, or choose from the predefined metrics defined in `<MRCMesh/MRMeshMetrics.h>`.
    MR_FillHoleMetric* metric = MR_getUniversalMetric( mesh );
    MR_FillHoleParams_Set_metric( params, MR_PassBy_Move, metric );
    MR_FillHoleMetric_Destroy( metric ); // `MR_PassBy_Move` is not destructive, the object still needs to be destroyed manually.
    // Optionally, receive the bitset of the created faces.
    MR_FaceBitSet* newFaces = MR_FaceBitSet_DefaultConstruct();
    MR_FillHoleParams_Set_outNewFaces( params, newFaces );

    // You can either fill all holes at once, or one by one.
    // In the latter case, don't forget to check the output fields of the parameters (e.g. `outNewFaces`) after every iteration.
    size_t newFaceCount = 0;
#define FILL_ALL_HOLES 1
#if FILL_ALL_HOLES
    MR_fillHoles( mesh, holes, params );
    newFaceCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( newFaces ) );
#else
    const float minHoleArea = 100.f; // An arbitrary size threshold for holes, just as a demonstration.
    for ( size_t i = 0; i < MR_std_vector_MR_EdgeId_Size( holes ); i++ )
    {
        MR_EdgeId e = *MR_std_vector_MR_EdgeId_At( holes, i );
        MR_Vector3d holeDirArea = MR_Mesh_holeDirArea( mesh, e );
        if ( MR_Vector3d_lengthSq( &holeDirArea ) >= minHoleArea*minHoleArea )
        {
            MR_fillHole( mesh, e, params );
            newFaceCount += MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( newFaces ) );
        }
    }
#endif

    printf( "Added %zu new faces\n", newFaceCount );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, output, NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_FaceBitSet_Destroy( newFaces );
fail_no_holes:
    MR_std_vector_MR_EdgeId_Destroy( holes );
fail_mesh_loading:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
