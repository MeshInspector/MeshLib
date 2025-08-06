#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBox.h>
#include <MRCMesh/MRICP.h>
#include <MRCMesh/MRMeshOrPoints.h>
#include <MRCMesh/MRMultiwayICP.h>
#include <MRCMesh/MRPointCloud.h>
#include <MRCMesh/MRPointCloudPart.h>
#include <MRCMesh/MRPointsLoad.h>
#include <MRCMesh/MRPointsSave.h>
#include <MRCMesh/MRString.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/expected_MR_PointCloud_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_function_bool_from_float.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// print progress every 10%
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

void resetProgress( void )
{
    gProgress = -1;
}

void printStats( const MR_MultiwayICP* icp )
{
    size_t numSamples = MR_MultiwayICP_getNumSamples( icp );
    size_t numActivePairs = MR_MultiwayICP_getNumActivePairs( icp );
    printf( "Samples: %zu\n", numSamples );
    printf( "Active point pairs: %zu\n", numActivePairs );
    if ( numActivePairs > 0 )
    {
        double p2ptMetric = MR_MultiwayICP_getMeanSqDistToPoint( icp, NULL );
        double p2ptInaccuracy = MR_MultiwayICP_getMeanSqDistToPoint( icp, &p2ptMetric );
        printf( "RMS point-to-point distance: %f ± %f\n", p2ptMetric, p2ptInaccuracy );

        double p2plMetric = MR_MultiwayICP_getMeanSqDistToPlane( icp, NULL );
        double p2plInaccuracy = MR_MultiwayICP_getMeanSqDistToPlane( icp, &p2plMetric );
        printf( "RMS point-to-plane distance: %f ± %f\n", p2plMetric, p2plInaccuracy );
    }
}

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    if ( argc < 4 )
    {
        fprintf( stderr, "Usage: %s INPUT1 INPUT2 [INPUTS...] OUTPUT\n", argv[0] );
        goto out;
    }

    // the global registration can be applied to meshes and point clouds
    // to simplify the sample app, we will work with point clouds only

    // Note that the default ICP method (point-to-plane) relies on the point cloud having the normals information, and will not work otherwise.
    // If your point cloud doesn't have normals, switch to the point-to-point method by calling `MR_ICPProperties_Set_method( params, MR_ICPMethod_PointToPoint );`.

    const int inputNum = argc - 2;

    // as ICP and MultiwayICP classes accept both meshes and point clouds,
    // the input data must be converted to special wrapper objects
    // NB: The wrapper is non-owning, it holds a reference to an existing mesh or point cloud and doesn't automatically destroy it when the wrapper is destroyed.
    //   So we manually destroy the wrapped contents before destroying the wrappers.
    MR_Vector_MR_MeshOrPointsXf_MR_ObjId* inputs = MR_Vector_MR_MeshOrPointsXf_MR_ObjId_DefaultConstruct();

    MR_Box3f maxBBox = MR_Box3f_DefaultConstruct();

    for ( int i = 0; i < inputNum; ++i )
    {
        MR_expected_MR_PointCloud_std_string* pointCloudEx = MR_PointsLoad_fromAnySupportedFormat_2( argv[1 + i], NULL, NULL );
        MR_PointCloud* pointCloud = MR_expected_MR_PointCloud_std_string_GetMutableValue( pointCloudEx );

        if ( !pointCloud )
        {
            // Failed to load.
            fprintf( stderr, "Failed to load point cloud: %s\n", MR_std_string_Data( MR_expected_MR_PointCloud_std_string_GetError( pointCloudEx ) ) );
            MR_expected_MR_PointCloud_std_string_Destroy( pointCloudEx );
            goto out_inputs;
        }

        // Construct the wrapper.
        MR_MeshOrPoints* mop = MR_MeshOrPoints_Construct_MR_PointCloud( pointCloud ); // Or `MR_MeshOrPoints_Construct_MR_Mesh()` for meshes.
        MR_MeshOrPointsXf* mopXf = MR_MeshOrPointsXf_ConstructFrom( mop, MR_AffineXf3f_DefaultConstruct() ); // Here you can optionally specify an affine transformation.
        MR_MeshOrPoints_Destroy( mop );

        // Insert the wrapper into the vector.
        MR_Vector_MR_MeshOrPointsXf_MR_ObjId_push_back_MR_MeshOrPointsXf_rvalue_ref( inputs, mopXf );
        MR_MeshOrPointsXf_Destroy( mopXf );

        MR_Box3f bbox = MR_PointCloud_computeBoundingBox_1( pointCloud, NULL );
        if ( !MR_Box3f_valid( &maxBBox ) || MR_Box3f_volume( &bbox ) > MR_Box3f_volume( &maxBBox ) )
            maxBBox = bbox;
    }

    // you can set various parameters for the global registration; see the documentation for more info
    MR_MultiwayICPSamplingParameters* samplingParams = MR_MultiwayICPSamplingParameters_DefaultConstruct();
    // set sampling voxel size
    MR_MultiwayICPSamplingParameters_Set_samplingVoxelSize( samplingParams, MR_Box3f_diagonal( &maxBBox ) * 0.03f );
    // set progress callback
    MR_std_function_bool_from_float* cb = MR_std_function_bool_from_float_DefaultConstruct();
    MR_std_function_bool_from_float_Assign( cb, onProgress );
    MR_MultiwayICPSamplingParameters_Set_cb( samplingParams, MR_PassBy_Copy, cb );

    MR_MultiwayICP* icp = MR_MultiwayICP_Construct( inputs, samplingParams );
    MR_MultiwayICPSamplingParameters_Destroy( samplingParams );

    MR_ICPProperties* params = MR_ICPProperties_DefaultConstruct();
    MR_MultiwayICP_setParams( icp, params );
    MR_ICPProperties_Destroy( params );

    // gather statistics
    MR_MultiwayICP_updateAllPointPairs( icp, MR_PassBy_DefaultArgument, NULL );
    printStats( icp );

    printf( "Calculating transformations...\n" );
    resetProgress();
    MR_Vector_MR_AffineXf3f_MR_ObjId* xfs = MR_MultiwayICP_calculateTransformations( icp, MR_PassBy_Copy, cb );
    printStats( icp );

    MR_PointCloud* output = MR_PointCloud_DefaultConstruct();
    for ( int i = 0; i < inputNum; i++ )
    {
        const MR_MeshOrPointsXf* input = MR_Vector_MR_MeshOrPointsXf_MR_ObjId_index_const( inputs, (MR_ObjId){i} );
        const MR_AffineXf3f* xf = MR_Vector_MR_AffineXf3f_MR_ObjId_index_const( xfs, (MR_ObjId){i} );
        const MR_PointCloud* cloud = MR_PointCloudPart_Get_cloud( MR_MeshOrPoints_asPointCloudPart( MR_MeshOrPointsXf_Get_obj( input ) ) );
        const MR_VertCoords* points = MR_PointCloud_Get_points( cloud );
        size_t numPoints = MR_VertCoords_size( points );
        printf("Resulting transform for part %d:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n", i, xf->A.x.x, xf->A.x.y, xf->A.x.z, xf->b.x, xf->A.y.x, xf->A.y.y, xf->A.y.z, xf->b.y, xf->A.z.x, xf->A.z.y, xf->A.z.z, xf->b.z);
        for ( size_t j = 0; j < numPoints; j++ )
        {
            MR_Vector3f point = *MR_VertCoords_index_const( points, (MR_VertId){j} );
            point = MR_AffineXf3f_call( xf, &point );
            MR_PointCloud_addPoint_1( output, &point );
        }
    }

    MR_expected_void_std_string* saveEx = MR_PointsSave_toAnySupportedFormat_3( output, argv[argc - 1], NULL, NULL );
    const MR_std_string* saveError = MR_expected_void_std_string_GetError( saveEx );
    if ( !saveError )
    {
        rc = EXIT_SUCCESS;
    }
    else
    {
        fprintf( stderr, "Failed to save point cloud: %s\n", MR_std_string_Data( saveError ) );
    }
    MR_std_string_Destroy( saveError );

    MR_PointCloud_Destroy( output );
    MR_Vector_MR_AffineXf3f_MR_ObjId_Destroy( xfs );
    MR_std_function_bool_from_float_Destroy( cb );
    MR_MultiwayICP_Destroy( icp );
out_inputs: ;
    // As explained earlier, since those wrappers are non-owning, we have to manually destroy the contents.
    size_t numLoadedInputs = MR_Vector_MR_MeshOrPointsXf_MR_ObjId_size( inputs );
    for ( size_t i = 0; i < numLoadedInputs; i++ )
    {
        MR_PointCloud_Destroy( MR_PointCloudPart_Get_cloud( MR_MeshOrPoints_asPointCloudPart( MR_MeshOrPointsXf_Get_obj( MR_Vector_MR_MeshOrPointsXf_MR_ObjId_index( inputs, (MR_ObjId){i} ) ) ) ) );
    }
    MR_Vector_MR_MeshOrPointsXf_MR_ObjId_Destroy( inputs );
out:
    return rc;
}
