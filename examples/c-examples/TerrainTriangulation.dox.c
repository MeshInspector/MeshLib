#include <MRCMesh/MRPointsLoad.h>
#include <MRCMesh/MRPointsLoadSettings.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRPointCloud.h>
#include <MRCMesh/MRTerrainTriangulation.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRSaveSettings.h>
#include <MRCMesh/MRColor.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_MR_PointCloud_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_vector_MR_Color.h>
#include <MRCMisc/std_vector_MR_Vector3f.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;
    // load points
    MR_VertColors* colors = MR_VertColors_DefaultConstruct();
    MR_PointsLoadSettings* pls = MR_PointsLoadSettings_ConstructFrom( colors, NULL, MR_PassBy_DefaultArgument, NULL );
    MR_expected_MR_PointCloud_std_string* loadRes = MR_PointsLoad_fromAnySupportedFormat_2( "logo.jpg", NULL, pls );
    MR_PointCloud* pc = MR_expected_MR_PointCloud_std_string_GetMutableValue( loadRes );
    if ( !pc )
    {
        fprintf( stderr, "Failed to load points: %s\n", MR_std_string_Data( MR_expected_MR_PointCloud_std_string_GetError( loadRes ) ) );
        goto fail_load; // error while loading file
    }

    MR_expected_MR_Mesh_std_string* triangulationRes = MR_terrainTriangulation(
        MR_PassBy_Copy,
        MR_VertCoords_GetMutable_vec_( MR_PointCloud_GetMutable_points( pc ) ),
        MR_PassBy_DefaultArgument, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( triangulationRes );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to triangulate points: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( triangulationRes ) ) );
        goto fail_triangulation; // error while triangulating
    }

    MR_SaveSettings* ss = MR_SaveSettings_DefaultConstruct();
    if ( MR_std_vector_MR_Color_Size( MR_VertColors_Get_vec_( colors ) ) ==
         MR_std_vector_MR_Vector3f_Size( MR_VertCoords_Get_vec_( MR_PointCloud_Get_points( pc ) ) )
         )
    {
        MR_SaveSettings_Set_colors( ss, colors );
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "TerrainMesh.ctm", NULL, ss );
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );

    MR_SaveSettings_Destroy( ss );
fail_triangulation:
    MR_expected_MR_Mesh_std_string_Destroy( triangulationRes );
fail_load:
    MR_expected_MR_PointCloud_std_string_Destroy( loadRes );

    MR_PointsLoadSettings_Destroy( pls );
    MR_VertColors_Destroy( colors );
    return rc;
}
