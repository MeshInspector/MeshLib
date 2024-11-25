#include "TestMacros.h"

#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshTopology.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRPointCloud.h"
#include "MRMeshC/MRPointCloudTriangulation.h"
#include "MRMeshC/MRVector3.h"

MRPointCloud* makeCube( void )
{
    MRPointCloud* pc = mrPointCloudNew();
    MRVector3f point = mrVector3fDiagonal( 0.0f );
    mrPointCloudAddPoint( pc, &point);

    point.x = 1.0f; point.y = 0.0f; point.z = 0.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 0.0f; point.y = 1.0f; point.z = 0.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 1.0f; point.y = 1.0f; point.z = 0.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 0.0f; point.y = 0.0f; point.z = 1.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 1.0f; point.y = 0.0f; point.z = 1.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 0.0f; point.y = 1.0f; point.z = 1.0f;
    mrPointCloudAddPoint( pc, &point );

    point.x = 1.0f; point.y = 1.0f; point.z = 1.0f;
    mrPointCloudAddPoint( pc, &point );

    return pc;
}

void testTriangulation( void )
{
    MRPointCloud* pc = makeCube();
    MRMesh* mesh = mrTriangulatePointCloud( pc, NULL );
    const size_t pointsCount = mrMeshPointsNum( mesh );
    TEST_ASSERT( pointsCount == 8 );
    const MRFaceBitSet* faces = mrMeshTopologyGetValidFaces( mrMeshTopology( mesh ) );
    const size_t count = mrBitSetCount( (MRBitSet*)faces );
    TEST_ASSERT( count > 0 );
    mrMeshFree( mesh );
    mrPointCloudFree( pc );
}
