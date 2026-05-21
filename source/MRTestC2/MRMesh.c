#include "TestMacros.h"

#include "MRMesh.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshPart.h"
#include "MRCMesh/MRCube.h"
#include "MRCMesh/MRTorus.h"
#include "MRCMesh/MRBitSet.h"
#include "MRCMesh/MRMeshFixer.h"
#include "MRCMesh/MRRegionBoundary.h"
#include "MRCMisc/expected_MR_UndirectedEdgeBitSet_std_string.h"

void testArea( void )
{
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );

    MR_Mesh* mesh = MR_makeCube( &size, &base );


    double area = MR_Mesh_area_const_MR_FaceBitSet_ptr( mesh, NULL );
    TEST_ASSERT( area > 5.999f && area < 6.001f );

    MR_FaceBitSet* faces = MR_FaceBitSet_DefaultConstruct();
    MR_FaceBitSet_resize( faces, 12, &(bool){true} );

    for ( int i = 6; i < 12; ++i )
        MR_FaceBitSet_set_2( faces, i, false );

    area = MR_Mesh_area_const_MR_FaceBitSet_ptr( mesh, faces );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    MR_Mesh_deleteFaces( mesh, faces, NULL );

    area = MR_Mesh_area_const_MR_FaceBitSet_ptr( mesh, NULL );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    MR_FaceBitSet_Destroy( faces );
    MR_Mesh_Destroy( mesh );
}

void testShortEdges( void )
{
    float primaryRadius = 1.f;
    float secondaryRadius = 0.05f;
    int primaryResolution = 16;
    int secondaryResolution = 16;
    MR_Mesh* mesh = MR_makeTorus(&primaryRadius, &secondaryRadius, &primaryResolution, &secondaryResolution, NULL);

    MR_MeshPart *mp = MR_MeshPart_Construct( mesh, NULL );

    MR_expected_MR_UndirectedEdgeBitSet_std_string* shortEdgesEx = MR_findShortEdges( mp, 0.1f, MR_PassBy_DefaultArgument, NULL );
    MR_UndirectedEdgeBitSet* shortEdges = MR_expected_MR_UndirectedEdgeBitSet_std_string_value_mut( shortEdgesEx );
    TEST_ASSERT( shortEdges );
    MR_MeshPart_Destroy( mp );

    size_t num = MR_UndirectedEdgeBitSet_count( shortEdges );
    TEST_ASSERT( num == 256 );
    MR_expected_MR_UndirectedEdgeBitSet_std_string_Destroy( shortEdgesEx );
    MR_Mesh_Destroy( mesh );
}

void testIncidentFacesFromVerts( void )
{
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );

    MR_Mesh* mesh = MR_makeCube( &size, &base );

    MR_VertBitSet* verts = MR_VertBitSet_DefaultConstruct();
    MR_VertBitSet_resize( verts, 8, &(bool){false} );
    MR_VertBitSet_set_2( verts, 0, true );

    MR_FaceBitSet* faces = MR_getIncidentFaces_MR_VertBitSet( MR_Mesh_Get_topology(mesh), verts );
    size_t num = MR_FaceBitSet_count( faces );
    TEST_ASSERT( num == 6 );
    MR_FaceBitSet_Destroy( faces );
    MR_VertBitSet_Destroy( verts );
    MR_Mesh_Destroy( mesh );
}

void testIncidentFacesFromEdges( void )
{
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );

    MR_Mesh* mesh = MR_makeCube( &size, &base );

    MR_UndirectedEdgeBitSet* edges = MR_UndirectedEdgeBitSet_DefaultConstruct();
    MR_UndirectedEdgeBitSet_resize( edges, 12, &(bool){false} );
    MR_UndirectedEdgeBitSet_set_2( edges, 0, &(bool){true} );

    MR_FaceBitSet* faces = MR_getIncidentFaces_MR_UndirectedEdgeBitSet( MR_Mesh_Get_topology( mesh ), edges );
    size_t num = MR_FaceBitSet_count( faces );
    TEST_ASSERT( num == 8 );
    MR_FaceBitSet_Destroy( faces );
    MR_UndirectedEdgeBitSet_Destroy( edges );
    MR_Mesh_Destroy( mesh );
}
