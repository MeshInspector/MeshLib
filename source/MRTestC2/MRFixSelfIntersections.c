#include "TestMacros.h"
#include "MRFixSelfIntersections.h"

#include "MRCMesh/MRFixSelfIntersections.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshTopology.h"
#include "MRCMesh/MRBitSet.h"
#include "MRCMesh/MRTorus.h"
#include "MRCMisc/expected_MR_FaceBitSet_std_string.h"
#include "MRCMisc/expected_void_std_string.h"

void testFixSelfIntersections( void )
{
    float primaryRadius = 1.0f;
    float secondaryRadius = 0.2f;
    int32_t primaryResolution = 32;
    int32_t secondaryResolution = 16;

    MR_Mesh* mesh = MR_makeTorusWithSelfIntersections( &primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL );
    size_t validFacesCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( mesh ) ) ) );
    TEST_ASSERT( validFacesCount == 1024 );

    MR_expected_MR_FaceBitSet_std_string* intersections_ex = MR_SelfIntersections_getFaces( mesh, false, MR_PassBy_DefaultArgument, NULL );
    MR_FaceBitSet* intersections = MR_expected_MR_FaceBitSet_std_string_GetMutableValue( intersections_ex );
    TEST_ASSERT( intersections );
    size_t intersectionsCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( intersections ) );
    TEST_ASSERT( intersectionsCount == 128 );
    MR_expected_MR_FaceBitSet_std_string_Destroy( intersections_ex );

    MR_SelfIntersections_Settings *settings = MR_SelfIntersections_Settings_DefaultConstruct();
    MR_SelfIntersections_Settings_Set_method( settings, MR_SelfIntersections_Settings_Method_CutAndFill );
    MR_SelfIntersections_Settings_Set_touchIsIntersection( settings, false );
    MR_expected_void_std_string_Destroy( MR_SelfIntersections_fix( mesh, settings ) );
    MR_SelfIntersections_Settings_Destroy( settings );

    validFacesCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( mesh ) ) ) );
    TEST_ASSERT( validFacesCount == 1194 );

    intersections_ex = MR_SelfIntersections_getFaces( mesh, false, MR_PassBy_DefaultArgument, NULL );
    intersections = MR_expected_MR_FaceBitSet_std_string_GetMutableValue( intersections_ex );
    TEST_ASSERT( intersections );
    intersectionsCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( intersections ) );
    TEST_ASSERT( intersectionsCount == 0 );
    MR_expected_MR_FaceBitSet_std_string_Destroy( intersections_ex );

    MR_Mesh_Destroy( mesh );
}
