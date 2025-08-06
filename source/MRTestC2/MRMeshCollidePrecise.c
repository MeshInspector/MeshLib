#include "TestMacros.h"

#include "MRMeshCollidePrecise.h"

#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRContoursCut.h>
#include <MRCMesh/MRIntersectionContour.h>
#include <MRCMesh/MRMatrix3.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshCollidePrecise.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MROneMeshContours.h>
#include <MRCMesh/MRPrecisePredicates3.h>
#include <MRCMesh/MRTorus.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/std_vector_MR_OneMeshContour.h>
#include <MRCMisc/std_vector_MR_OneMeshIntersection.h>
#include <MRCMisc/std_vector_MR_VarEdgeTri.h>
#include <MRCMisc/std_vector_std_vector_MR_VarEdgeTri.h>

void testMeshCollidePrecise( void )
{
    float primaryRadius = 1.1f;
    float secondaryRadius = 0.5f;
    int32_t primaryResolution = 8;
    int32_t secondaryResolution = 8;
    MR_Mesh* meshA = MR_makeTorus( &primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL );
    MR_Mesh* meshB = MR_makeTorus( &primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL );

    MR_Vector3f from = MR_Vector3f_plusZ();
    MR_Vector3f to = { 0.1f, 0.8f, 0.2f };
    MR_Matrix3f rot = MR_Matrix3f_rotation_MR_Vector3f( &from, &to );
    MR_AffineXf3f xf = MR_AffineXf3f_linear( &rot );
    MR_Mesh_transform( meshB, &xf, NULL );

    MR_MeshPart* meshAPart = MR_MeshPart_Construct( meshA, NULL );
    MR_MeshPart* meshBPart = MR_MeshPart_Construct( meshB, NULL );
    MR_CoordinateConverters* conv = MR_getVectorConverters_3( meshAPart, meshBPart, NULL );

    MR_std_vector_MR_VarEdgeTri* intersections = MR_findCollidingEdgeTrisPrecise_5( meshAPart, meshBPart, MR_PassBy_Copy, MR_CoordinateConverters_GetMutable_toInt( conv ), NULL, false );

    MR_MeshPart_Destroy( meshAPart );
    MR_MeshPart_Destroy( meshBPart );

    TEST_ASSERT_INT_EQUAL( (int)MR_std_vector_MR_VarEdgeTri_Size( intersections ), 152 )

    const MR_MeshTopology* meshATop = MR_Mesh_Get_topology( meshA );
    const MR_MeshTopology* meshBTop = MR_Mesh_Get_topology( meshB );
    MR_std_vector_std_vector_MR_VarEdgeTri* contours = MR_orderIntersectionContours( meshATop, meshBTop, intersections );
    TEST_ASSERT_INT_EQUAL( (int)MR_std_vector_std_vector_MR_VarEdgeTri_Size( contours ), 4 )
    TEST_ASSERT_INT_EQUAL( (int)MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 0 ) ), 71 )
    TEST_ASSERT_INT_EQUAL( (int)MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 1 ) ), 7 )
    TEST_ASSERT( MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 2 ) ) == 69 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 2 ) ) == 71 ); // with FMA instruction (modern compilers for ARM)
    TEST_ASSERT( MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 3 ) ) == 9 ||  // without FMA instruction (default settings for x86 or old compilers for ARM)
                 MR_std_vector_MR_VarEdgeTri_Size( MR_std_vector_std_vector_MR_VarEdgeTri_At( contours, 3 ) ) == 7 );  // with FMA instruction (modern compilers for ARM)

    MR_std_vector_MR_OneMeshContour* meshAContours = MR_std_vector_MR_OneMeshContour_DefaultConstruct();
    MR_std_vector_MR_OneMeshContour* meshBContours = MR_std_vector_MR_OneMeshContour_DefaultConstruct();
    MR_getOneMeshIntersectionContours( meshA, meshB, contours, meshAContours, meshBContours, conv, NULL, NULL, NULL );
    TEST_ASSERT( MR_std_vector_MR_OneMeshContour_Size( meshAContours ) == 4 )
    TEST_ASSERT( MR_std_vector_MR_OneMeshContour_Size( meshBContours ) == 4 )

    size_t posCount = 0;
    for ( size_t i = 0; i < MR_std_vector_MR_OneMeshContour_Size( meshAContours ); ++i )
        posCount += MR_std_vector_MR_OneMeshIntersection_Size( MR_OneMeshContour_Get_intersections( MR_std_vector_MR_OneMeshContour_At( meshAContours, i ) ) );
    TEST_ASSERT( posCount == 156 )

    MR_std_vector_MR_OneMeshContour_Destroy( meshBContours );
    MR_std_vector_MR_OneMeshContour_Destroy( meshAContours );

    MR_std_vector_std_vector_MR_VarEdgeTri_Destroy( contours );

    MR_std_vector_MR_VarEdgeTri_Destroy( intersections );

    MR_CoordinateConverters_Destroy( conv );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}
