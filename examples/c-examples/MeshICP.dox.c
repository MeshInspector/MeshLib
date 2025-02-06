#include <MRMeshC/MRICP.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    // Load meshes
    MRMesh* meshFloating = mrMeshLoadFromAnySupportedFormat( "meshA.stl", NULL );
    MRMesh* meshFixed = mrMeshLoadFromAnySupportedFormat( "meshB.stl", NULL );

    // Prepare ICP parameters
    MRBox3f bbox = mrMeshComputeBoundingBox( meshFixed, NULL );
    float diagonal = mrBox3fDiagonal( &bbox );
    float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
    MRICPProperties icpParams = mrICPPropertiesNew();
    icpParams.distThresholdSq = diagonal * diagonal * 0.01f; // Use points pairs with maximum distance specified
    icpParams.exitVal = diagonal * 0.003f; // Stop when distance reached

    // Calculate transformation
    MRMeshOrPointsXf* flt = mrMeshOrPointsXfFromMesh( meshFloating, NULL );
    MRMeshOrPointsXf* ref = mrMeshOrPointsXfFromMesh( meshFixed, NULL );
    MRICP* icp = mrICPNew( flt, ref, icpSamplingVoxelSize );
    mrICPSetParams( icp, &icpParams );
    MRAffineXf3f xf = mrICPCalculateTransformation( icp );

    // Transform floating mesh
    mrMeshTransform( meshFloating, &xf, NULL );

    // Output information string
    MRString* info = mrICPGetStatusInfo( icp );
    printf( "%s\n", mrStringData( info ) );

    // Save result
    mrMeshSaveToAnySupportedFormat( meshFloating, "meshA_icp.stl", NULL, NULL );

    mrStringFree( info );
    mrICPFree( icp );
    mrMeshOrPointsXfFree( flt );
    mrMeshOrPointsXfFree( ref );
    mrMeshFree( meshFixed );
    mrMeshFree( meshFloating );
    return EXIT_SUCCESS;
}
