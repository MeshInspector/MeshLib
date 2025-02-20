#include <stdio.h>
#include "TestMacros.h"
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
    RUN_TEST( testAddNoise )
    RUN_TEST( testArea )
    RUN_TEST( testBoxi )
    RUN_TEST( testBoxf )
    RUN_TEST( testBoxfInvalid )
    RUN_TEST( testBoxiInvalid )
    RUN_TEST( testBitSet )
    RUN_TEST( testDegenerateBandNonEmpty )
    RUN_TEST( testDegenerateBandEmpty )
    RUN_TEST( testMeshBoolean )
    RUN_TEST( testOffsetMesh )
    RUN_TEST( testDoubleOffsetMesh )
    RUN_TEST( testMcOffsetMesh )
    RUN_TEST( testSharpOffsetMesh )
    RUN_TEST( testGeneralOffsetMesh )
    RUN_TEST( testThickenMesh )
    RUN_TEST( testMeshSubdivide )
    RUN_TEST( testBooleanMultipleEdgePropogationSort )
    RUN_TEST( testBooleanMapper )
    RUN_TEST( testMeshCollidePrecise )
    RUN_TEST( testMeshDecimate )
    RUN_TEST( testMeshFillHole )
    RUN_TEST( testMeshFillHoleNicely )
    RUN_TEST( testMeshNormals )
    RUN_TEST( testComponentsMap )
    RUN_TEST( testLargeRegions )
    RUN_TEST( testUniteCloseVertices )
    RUN_TEST( testLargeComponents )
    RUN_TEST( testLargestComponent )
    RUN_TEST( testGetComponent )
    RUN_TEST( testFixSelfIntersections )
    RUN_TEST( testRightBoundary )
    RUN_TEST( testFindHoleComplicatingFaces )
    RUN_TEST( testExpandShrink )
    RUN_TEST( testExpandShrinkVerts )
    RUN_TEST( testShortEdges )
    RUN_TEST( testIncidentFacesFromVerts )
    RUN_TEST( testIncidentFacesFromEdges )
    RUN_TEST( testTriangulation )
    RUN_TEST( testVDBConversions )
    RUN_TEST( testUniformResampling )
    RUN_TEST( testResampling )
    RUN_TEST( testCropping )
    RUN_TEST( testAccessors )
     //MRMeshTopology
    RUN_TEST( testMrMeshTopologyPack )
    RUN_TEST( testMrMeshTopologyGetValidVerts )
    RUN_TEST( testMrMeshTopologyGetValidFaces )
    RUN_TEST( testMrMeshTopologyFindHoleRepresentiveEdges )
    RUN_TEST( testMrMeshTopologyGetLeftTriVerts )
    RUN_TEST( testMrMeshTopologyFindNumHoles )
    RUN_TEST( testMrMeshTopologyFaceSize )
    RUN_TEST( testMrMeshTopologyGetTriangulation )
      
    printf("Tests finished\n");
}
