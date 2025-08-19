#include "TestMacros.h"
#include "MRFixSelfIntersections.h"

#include "MRMeshC/MRFixSelfIntersections.h"
#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshTopology.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRTorus.h"

void testFixSelfIntersections( void )
{
    MRMakeTorusParameters params;
    params.primaryRadius = 1.0f;
    params.secondaryRadius = 0.2f;
    params.primaryResolution = 32;
    params.secondaryResolution = 16;

    MRString* error = NULL;
    MRMesh* mesh = mrMakeTorusWithSelfIntersections( &params );
    size_t validFacesCount = mrBitSetCount( ( MRBitSet* ) mrMeshTopologyGetValidFaces( mrMeshTopology( mesh ) ) );
    TEST_ASSERT( validFacesCount == 1024 );

    MRFaceBitSet* intersections = mrFixSelfIntersectionsGetFaces( mesh, false, NULL, &error );
    TEST_ASSERT( !error );
    size_t intersectionsCount = mrBitSetCount( ( MRBitSet* ) intersections );
    TEST_ASSERT( intersectionsCount == 128 );
    mrFaceBitSetFree( intersections );

    MRFixSelfIntersectionsSettings settings = mrFixSelfIntersectionsSettingsNew();
    settings.method = MRFixSelfIntersectionsMethodCutAndFill;
    settings.touchIsIntersection = false;
    mrFixSelfIntersectionsFix( mesh, &settings, NULL );

    validFacesCount = mrBitSetCount( ( MRBitSet* ) mrMeshTopologyGetValidFaces( mrMeshTopology( mesh ) ) );
    TEST_ASSERT( validFacesCount == 1194 );

    intersections = mrFixSelfIntersectionsGetFaces( mesh, false, NULL, &error );
    TEST_ASSERT( !error );
    intersectionsCount = mrBitSetCount( ( MRBitSet* ) intersections );
    TEST_ASSERT( intersectionsCount == 0 );
    mrFaceBitSetFree( intersections );

    mrMeshFree( mesh );
}
