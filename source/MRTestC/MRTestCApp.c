#include <stdio.h>
#include "MRAddNoise.h"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshDecimate.h"
#include "MRMeshExtrude.h"
#include "MRMeshFillHole.h"
#include "MRMeshNormals.h"
#include "MRMeshOffset.h"
#include "MRMeshComponents.h"
#include "MRMeshTopology.h"
#include "MRMeshBuilder.h"
#include "MRMeshSubdivide.h"
#include "MRFixSelfIntersections.h"
#include "MRExpandShrink.h"
#include "MRPointCloud.h"
#include "MRVDBConversions.h"

int main( void )
{
    testAddNoise();
    testArea();
    testBoxi();
    testBoxf();
    testBoxfInvalid();
    testBoxiInvalid();
    testBitSet();
    testDegenerateBandNonEmpty();
    testDegenerateBandEmpty();
    testMeshBoolean();
    testOffsetMesh();
    testDoubleOffsetMesh();
    testMcOffsetMesh();
    testSharpOffsetMesh();
    testGeneralOffsetMesh();
    testThickenMesh();
    testMeshSubdivide();
    testBooleanMultipleEdgePropogationSort();
    testBooleanMapper();
    testMeshCollidePrecise();
    testMeshDecimate();
    testMeshFillHole();
    testMeshFillHoleNicely();
    testMeshNormals();
    testComponentsMap();
    testLargeRegions();
    testUniteCloseVertices();
    testLargeComponents();
    testLargestComponent();
    testGetComponent();
    testFixSelfIntersections();
    testRightBoundary();
    testFindHoleComplicatingFaces();
    testExpandShrink();
    testExpandShrinkVerts();
    testShortEdges();
    testIncidentFacesFromVerts();
    testIncidentFacesFromEdges();
    testTriangulation();
    testVDBConversions();
    testUniformResampling();
    testResampling();
    testCropping();
    testAccessors();

   //MRMeshToplogy
    test_mrMeshTopologyPack();
    test_mrMeshTopologyGetValidVerts();
    test_mrMeshTopologyGetValidFaces();
    test_mrMeshTopologyFindHoleRepresentiveEdges();
    test_mrMeshTopologyGetLeftTriVerts();
    test_mrMeshTopologyFindNumHoles();
    test_mrMeshTopologyFaceSize();
    test_mrMeshTopologyGetTriangulation();

    printf("Tests finished\n");
}
