#include <MRCMesh/MRImageLoad.h>
#include <MRCMesh/MRDistanceMap.h>
#include <MRCMesh/MRPolyline.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MR2DContoursTriangulation.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_MR_Image_std_string.h>
#include <MRCMisc/expected_MR_DistanceMap_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_vector_std_vector_MR_Vector2f.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;
    // load image as distance map
    MR_expected_MR_Image_std_string* imageLoadRes = MR_ImageLoad_fromAnySupportedFormat( "logo.jpg", NULL );
    MR_Image* image = MR_expected_MR_Image_std_string_GetMutableValue( imageLoadRes );
    if ( !image )
    {
        fprintf( stderr, "Failed to load image: %s\n", MR_std_string_Data( MR_expected_MR_Image_std_string_GetError( imageLoadRes ) ) );
        goto fail_load; // error while loading file
    }

    MR_expected_MR_DistanceMap_std_string* dmRes = MR_convertImageToDistanceMap( image, NULL, NULL );
    MR_DistanceMap* dm = MR_expected_MR_DistanceMap_std_string_GetMutableValue( dmRes );
    if ( !dm )
    {
        fprintf( stderr, "Failed to convert image to distance map: %s\n", MR_std_string_Data( MR_expected_MR_DistanceMap_std_string_GetError( dmRes ) ) );
        goto fail_conversion; // error while converting image to distance map
    }

    MR_Polyline2* pl2 = MR_distanceMapTo2DIsoPolyline_2( dm, 210.0f );
    MR_std_vector_std_vector_MR_Vector2f* conts = MR_Polyline2_contours( pl2, NULL );

    MR_Mesh* triangulationRes = MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2f( conts, NULL );

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( triangulationRes, "LogoMesh.ctm", NULL, NULL );
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );

    MR_Mesh_Destroy( triangulationRes );
    MR_std_vector_std_vector_MR_Vector2f_Destroy( conts );
    MR_Polyline2_Destroy( pl2 );
fail_conversion:
    MR_expected_MR_DistanceMap_std_string_Destroy( dmRes );
fail_load:
    MR_expected_MR_Image_std_string_Destroy( imageLoadRes );
    return rc;
}
