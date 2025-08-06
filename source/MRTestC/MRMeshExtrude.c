#include "TestMacros.h"

#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshExtrude.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRTorus.h"
#include <stdlib.h>

// Test for the case when the region contains selected faces
void testDegenerateBandNonEmpty(void) {
    // Create a torus mesh
    MRMakeTorusParameters params = { 1.1f, 0.5f, 16, 16 };
    MRMesh* mesh = mrMakeTorus(&params);
    TEST_ASSERT(mesh != NULL);

    // Record the number of vertices before calling the function
    size_t vCountBefore = mrMeshPointsNum(mesh);

    // Create a face bit set and select a few faces (for example, faces 0 and 1)
    MRFaceBitSet* region = mrFaceBitSetNew(0, false);
    mrBitSetAutoResizeSet((MRBitSet*)region, 0, true);
    mrBitSetAutoResizeSet((MRBitSet*)region, 1, true);

    // Call the function to create a band of degenerate faces along the region boundary
    mrMakeDegenerateBandAroundRegion(mesh, region);

    // Invalidate caches
    mrMeshInvalidateCaches(mesh, true);

    // Get the number of vertices after the function call
    size_t vCountAfter = mrMeshPointsNum(mesh);

    // Expect the number of vertices to increase due to vertex duplication along the boundary
    TEST_ASSERT( vCountBefore - vCountAfter != 0);

    mrFaceBitSetFree(region);
    mrMeshFree(mesh);
}

// Test for the case when the region is empty (no selected faces)
void testDegenerateBandEmpty(void) {
    // Create a torus mesh
    MRMakeTorusParameters params = { 1.1f, 0.5f, 16, 16 };
    MRMesh* mesh = mrMakeTorus(&params);
    TEST_ASSERT(mesh != NULL);

    // Record the number of vertices before calling the function
    size_t vCountBefore = mrMeshPointsNum(mesh);

    // Create an empty face bit set
    MRFaceBitSet* region = mrFaceBitSetNew(0, false);

    // Call the function with an empty region, no changes should occur
    mrMakeDegenerateBandAroundRegion(mesh, region);
    mrMeshInvalidateCaches(mesh, true);

    // The number of vertices should remain the same
    size_t vCountAfter = mrMeshPointsNum(mesh);
    TEST_ASSERT(vCountAfter == vCountBefore);

    mrFaceBitSetFree(region);
    mrMeshFree(mesh);
}
