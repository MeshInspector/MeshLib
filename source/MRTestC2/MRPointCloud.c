#include "TestMacros.h"

#include "MRCMesh/MRBitSet.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshToPointCloud.h"
#include "MRCMesh/MRMeshTopology.h"
#include "MRCMesh/MRPointCloud.h"
#include "MRCMesh/MRPointCloudTriangulation.h"
#include "MRCMesh/MRTorus.h"
#include "MRCMesh/MRVector.h"
#include "MRCMesh/MRVector3.h"
#include "MRCMisc/std_optional_MR_Mesh.h"
#include "MRCMisc/std_vector_MR_EdgeId.h"
#include "MRMesh.h"
#include "MRPointCloud.h"

void testTriangulation( void )
{
    float primaryRadius = 2.f;
    float secondaryRadius = 1.0f;
    int32_t primaryResolution = 32;
    int32_t secondaryResolution = 32;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    MR_PointCloud* pc = MR_meshToPointCloud( mesh, &(bool){true}, NULL );
    MR_std_optional_MR_Mesh* restored = MR_triangulatePointCloud( pc, NULL, MR_PassBy_DefaultArgument, NULL);
    TEST_ASSERT( MR_VertCoords_size( MR_Mesh_Get_points( MR_std_optional_MR_Mesh_Value( restored ) ) ) == 1024 );
    const MR_MeshTopology* top = MR_Mesh_Get_topology( MR_std_optional_MR_Mesh_Value( restored ) );
    TEST_ASSERT( MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( MR_MeshTopology_getValidVerts( top ) ) ) == 1024 );
    MR_std_vector_MR_EdgeId* holes = MR_MeshTopology_findHoleRepresentiveEdges( top, NULL );

    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( holes ) == 0 );

    MR_std_vector_MR_EdgeId_Destroy( holes );
    MR_Mesh_Destroy( mesh );
    MR_std_optional_MR_Mesh_Destroy( restored );
    MR_PointCloud_Destroy( pc );
}
