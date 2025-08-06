#include "TestMacros.h"

#include "MRAddNoise.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRColor.h"
#include "MRExpandShrink.h"
#include "MRFixSelfIntersections.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRMeshBuilder.h"
#include "MRMeshCollide.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshComponents.h"
#include "MRMeshDecimate.h"
#include "MRMeshExtrude.h"
#include "MRMeshFillHole.h"
#include "MRMeshMeshDistance.h"
#include "MRMeshNormals.h"
#include "MRMeshOffset.h"
#include "MRMeshSubdivide.h"
#include "MRMeshTopology.h"
#include "MRPointCloud.h"
#include "MRPointsToMeshProjector.h"
#include "MRVDBConversions.h"
#include "MRVector3.h"
#include "SimpleObjects.h"

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
    RUN_TEST( testMeshCollide )
    RUN_TEST( testMeshCollidePrecise )
    RUN_TEST( testMeshDecimate )
    RUN_TEST( testMeshFillHole )
    RUN_TEST( testMeshFillHoleNicely )
    RUN_TEST( testMeshMeshDistance )
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

    // MRColor
    RUN_TEST( testMrColorNew )
    RUN_TEST( testMrColorFromComponents )
    RUN_TEST( testMrColorFromFloatComponents )
    RUN_TEST( testMrColorGetUInt32 )
    RUN_TEST( testMrVertColorsNewSized )

    // MRMatrix3

    RUN_TEST( testMrMatrix3fIdentity )
    RUN_TEST( testMrMatrix3fRotationScalar )
    RUN_TEST( testMrMatrix3fRotationVector )
    RUN_TEST( testMrMatrix3fAdd )
    RUN_TEST( testMrMatrix3fSub )
    RUN_TEST( testMrMatrix3fMul )
    RUN_TEST( testMrMatrix3fMulVector )
    RUN_TEST( testMrMatrix3fEqual )

    // MRAffineXf

    RUN_TEST( testMrAffineXf3fNew );
    RUN_TEST( testMrAffineXf3fTranslation );
    RUN_TEST( testMrAffineXf3fLinear );
    RUN_TEST( testMrAffineXf3fMul );
    RUN_TEST( testMrAffineXf3fApply );

    // MRSimpleObjects
    RUN_TEST( testMrMakeCube );
    RUN_TEST( testMrMakeCylinderAdvanced );
    RUN_TEST( testMrMakeTorus );
    RUN_TEST( testMrMakeTorusWithSelfIntersections );
    RUN_TEST( testMrMakeSphere );
    RUN_TEST( testMrMakeUVSphere );

    //MRVector
    RUN_TEST( testMrVector3fDiagonal );
    RUN_TEST( testMrVector3fPlusX );
    RUN_TEST( testMrVector3fPlusY );
    RUN_TEST( testMrVector3fPlusZ );
    RUN_TEST( testMrVector3fAdd );
    RUN_TEST( testMrVector3fSub );
    RUN_TEST( testMrVector3fMulScalar );
    RUN_TEST( testMrVector3fLengthSq );
    RUN_TEST( testMrVector3fLength );
    RUN_TEST( testMrVector3iDiagonal );
    RUN_TEST( testMrVector3iPlusX );
    RUN_TEST( testMrVector3iPlusY );
    RUN_TEST( testMrVector3iPlusZ );

    // MRPointsToMeshProjector
    RUN_TEST( testFindSignedDistances );

    printf("Tests finished\n");
}
