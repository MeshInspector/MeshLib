#include "TestMacros.h"

#include "MRPointsToMeshProjector.h"

#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRPointsToMeshProjector.h"

void testFindSignedDistances( void )
{
    MRMesh* meshA = createCube();
    MRMesh* meshB = createCube();

    MRMeshProjectionParameters params = mrMeshProjectionParametersNew();
    const MRVector3f shift = { 1.f, 2.f, 3.f };
    const MRAffineXf3f xf = mrAffineXf3fTranslation( &shift );
    params.xf = &xf;

    MRScalars* results = mrFindSignedDistances( meshA, meshB, &params );
    TEST_ASSERT_INT_EQUAL( (int)results->size, (int)mrMeshPointsNum( meshB ) );

    float maxDist = 0.f;
    for ( int i = 0; i < results->size; i++ )
        if ( maxDist < fabsf( results->data[i] ) )
            maxDist = fabsf( results->data[i] );
    TEST_ASSERT_FLOAT_EQUAL_APPROX( maxDist, mrVector3fLength( &shift ), 1e-6f )

    mrScalarsFree( results );

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}
