#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBox.h>
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRString.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>
#include <MRCVoxels/MROffset.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define APPROX_VOXEL_COUNT 10000000.f

int main( void )
{
    int rc = EXIT_FAILURE;

    // Create mesh
    MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );
    MR_Mesh* mesh = MR_makeCube( &size, &base );

    // offset functions can also be applied to separate mesh components rather than to the whole mesh
    // this is not our case, so the region is set to NULL
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct( mesh, NULL );

    // Setup parameters
    MR_OffsetParameters* params = MR_OffsetParameters_DefaultConstruct();
    // calculate voxel size depending on desired accuracy and/or memory consumption
    MR_BaseShellParameters_Set_voxelSize( MR_OffsetParameters_MutableUpcastTo_MR_BaseShellParameters( params ), MR_suggestVoxelSize( inputMeshPart, 10000000.f ) );
    MR_Box3f bbox = MR_Mesh_computeBoundingBox_1( mesh, NULL );
    float offset = MR_Box3f_diagonal( &bbox ) * 0.1f;

    // Make offset mesh
    MR_expected_MR_Mesh_std_string* outputMeshEx = MR_offsetMesh( inputMeshPart, offset, params );
    MR_MeshPart_Destroy( inputMeshPart );
    MR_OffsetParameters_Destroy( params );

    MR_Mesh* outputMesh = MR_expected_MR_Mesh_std_string_GetMutableValue( outputMeshEx );

    if ( !outputMesh )
    {
        fprintf( stderr, "Failed to perform offset: %s", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( outputMeshEx ) ) );
        goto fail_offset;
    }

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( outputMesh, "mesh_offset.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_offset:
    MR_expected_MR_Mesh_std_string_Destroy( outputMeshEx );
    MR_Mesh_Destroy( mesh );
    return rc;
}
