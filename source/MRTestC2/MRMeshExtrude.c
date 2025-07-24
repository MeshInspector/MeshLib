#include "TestMacros.h"

#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshExtrude.h>
#include <MRCMesh/MRTorus.h>
#include <MRCMesh/MRVector.h>

#include <stdlib.h>

// Test for the case when the region contains selected faces
void testDegenerateBandNonEmpty(void) {
    // Create a torus mesh
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 16;
    int32_t secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);
    TEST_ASSERT(mesh != NULL);

    // Record the number of vertices before calling the function
    size_t vCountBefore = MR_VertCoords_size( MR_Mesh_Get_points( mesh ) );

    // Create a face bit set and select a few faces (for example, faces 0 and 1)
    MR_FaceBitSet* region = MR_FaceBitSet_DefaultConstruct();

    MR_BitSet_autoResizeSet_2( MR_FaceBitSet_MutableUpcastTo_MR_BitSet( region ), 0, &(bool){true} );
    MR_BitSet_autoResizeSet_2( MR_FaceBitSet_MutableUpcastTo_MR_BitSet( region ), 1, &(bool){true} );

    // Call the function to create a band of degenerate faces along the region boundary
    MR_makeDegenerateBandAroundRegion( mesh, region, NULL );

    // Invalidate caches
    MR_Mesh_invalidateCaches( mesh, &(bool){true} );

    // Get the number of vertices after the function call
    size_t vCountAfter = MR_VertCoords_size( MR_Mesh_Get_points( mesh ) );

    // Expect the number of vertices to increase due to vertex duplication along the boundary
    TEST_ASSERT( vCountBefore - vCountAfter != 0);

    MR_FaceBitSet_Destroy(region);
    MR_Mesh_Destroy(mesh);
}

// Test for the case when the region is empty (no selected faces)
void testDegenerateBandEmpty(void) {
    // Create a torus mesh
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 16;
    int32_t secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);
    TEST_ASSERT(mesh != NULL);

    // Record the number of vertices before calling the function
    size_t vCountBefore = MR_VertCoords_size( MR_Mesh_Get_points( mesh ) );

    // Create an empty face bit set
    MR_FaceBitSet* region = MR_FaceBitSet_DefaultConstruct();

    // Call the function with an empty region, no changes should occur
    MR_makeDegenerateBandAroundRegion( mesh, region, NULL);
    MR_Mesh_invalidateCaches( mesh, &(bool){true} );

    // The number of vertices should remain the same
    size_t vCountAfter = MR_VertCoords_size( MR_Mesh_Get_points( mesh ) );
    TEST_ASSERT(vCountAfter == vCountBefore);

    MR_FaceBitSet_Destroy(region);
    MR_Mesh_Destroy(mesh);
}
