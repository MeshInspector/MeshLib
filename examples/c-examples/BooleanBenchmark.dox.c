#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBooleanOperation.h>
#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMatrix3.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBoolean.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/std_string.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA.
// sphereB itself is not modified, so the AABB trees of both meshes are built once and reused.
static bool subtractRotated( const MR_Mesh* sphereA, const MR_Mesh* sphereB, int i )
{
    MR_Vector3f plusZ = MR_Vector3f_plusZ();
    MR_Matrix3f rotation = MR_Matrix3f_rotation_float( &plusZ, i * 0.1f * 3.14159265f / 180 );
    MR_AffineXf3f rigidB2A = MR_AffineXf3f_linear( &rotation );

    MR_BooleanResult* result = MR_boolean_withXf( sphereA, sphereB, MR_BooleanOperation_DifferenceAB, &rigidB2A, NULL, MR_PassBy_DefaultArgument, NULL );
    bool valid = MR_BooleanResult_valid( result );
    if ( !valid )
        fprintf( stderr, "Failed to perform boolean: %s\n", MR_std_string_data( MR_BooleanResult_Get_errorString( result ) ) );
    MR_BooleanResult_Destroy( result );
    return valid;
}

static double nowSeconds( void )
{
    struct timespec ts;
    timespec_get( &ts, TIME_UTC );
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;

    // The number of Booleans can be given in the command line.
    int numBooleans = argc > 1 ? atoi( argv[1] ) : 2000;

    // Two identical spheres of radius 1.
    MR_SphereParams* params = MR_SphereParams_ConstructFrom( 1.f, 3366 );
    MR_Mesh* sphereA = MR_makeSphere( params );
    MR_Mesh* sphereB = MR_Mesh_ConstructFromAnother( MR_PassBy_Copy, sphereA );
    MR_SphereParams_Destroy( params );

    // Untimed warm-up Boolean, it builds the AABB trees.
    if ( !subtractRotated( sphereA, sphereB, 1 ) )
        goto fail;

    double start = nowSeconds();
    for ( int i = 1; i <= numBooleans; ++i )
        if ( !subtractRotated( sphereA, sphereB, i ) )
            goto fail;
    double seconds = nowSeconds() - start;

    printf( "%d Booleans in %.3f s, %.1f Booleans/s\n", numBooleans, seconds, numBooleans / seconds );

    rc = EXIT_SUCCESS;
fail:
    MR_Mesh_Destroy( sphereB );
    MR_Mesh_Destroy( sphereA );
    return rc;
}
