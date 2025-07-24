#include "TestMacros.h"

#include "MRMeshFillHole.h"

#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRFillHoleNicely.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshFillHole.h>
#include <MRCMesh/MRMeshFixer.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRRegionBoundary.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/std_vector_MR_EdgeId.h>
#include <MRCMisc/std_vector_MR_Vector3f.h>
#include <MRCMisc/std_vector_std_vector_MR_EdgeId.h>

MR_Mesh* createMeshWithHoles( void )
{
    MR_VertCoords* pointsVec = MR_VertCoords_Construct_1_uint64_t( 6 );
    *MR_VertCoords_index( pointsVec, (MR_VertId){0} ) = (MR_Vector3f){ 0.f, 0.f, 0.f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){1} ) = (MR_Vector3f){ 1.f, 0.f, 0.f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){2} ) = (MR_Vector3f){ 0.f, 1.f, 0.f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){3} ) = (MR_Vector3f){ 0.f, 0.f, 1.f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){4} ) = (MR_Vector3f){ 1.f, 0.f, 1.f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){5} ) = (MR_Vector3f){ 0.f, 1.f, 1.f };

    MR_Triangulation* triangulation = MR_Triangulation_Construct_1_uint64_t( 5 );
    *MR_Triangulation_index( triangulation, (MR_FaceId){0} ) = (MR_std_array_MR_VertId_3){{ {0}, {2}, {1} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){1} ) = (MR_std_array_MR_VertId_3){{ {3}, {4}, {5} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){2} ) = (MR_std_array_MR_VertId_3){{ {0}, {1}, {3} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){3} ) = (MR_std_array_MR_VertId_3){{ {2}, {5}, {4} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){4} ) = (MR_std_array_MR_VertId_3){{ {2}, {3}, {5} }};

    MR_Mesh* ret = MR_Mesh_fromTriangles( MR_PassBy_Copy, pointsVec, triangulation, NULL, MR_PassBy_DefaultArgument, NULL );

    MR_Triangulation_Destroy( triangulation );
    MR_VertCoords_Destroy( pointsVec );

    return ret;
}

void testMeshFillHole( void )
{
    MR_Mesh* mesh = createMeshWithHoles();

    MR_std_vector_MR_EdgeId* oldBdEdges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( oldBdEdges ) == 2 );

    MR_FillHoleParams* params = MR_FillHoleParams_DefaultConstruct();
    MR_FaceBitSet* newFaces = MR_FaceBitSet_DefaultConstruct();
    MR_FillHoleParams_Set_outNewFaces( params, newFaces );
    MR_fillHoles( mesh, oldBdEdges, params );
    MR_FillHoleParams_Destroy( params );

    TEST_ASSERT( MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( newFaces ) ) == 3 )

    MR_std_vector_MR_EdgeId* newBdEdges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( newBdEdges ) == 0 );

    MR_std_vector_MR_EdgeId_Destroy( newBdEdges );
    MR_FaceBitSet_Destroy( newFaces );
    MR_std_vector_MR_EdgeId_Destroy( oldBdEdges );
    MR_Mesh_Destroy( mesh );
}

void testMeshFillHoleNicely( void )
{
    MR_Mesh* mesh = createMeshWithHoles();
    MR_std_vector_MR_EdgeId* oldBdEdges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( oldBdEdges ) == 2 );

    MR_FillHoleNicelySettings* params = MR_FillHoleNicelySettings_DefaultConstruct();

    MR_FaceBitSet* patch = MR_fillHoleNicely( mesh, *MR_std_vector_MR_EdgeId_Front( oldBdEdges ), params );
    MR_FillHoleNicelySettings_Destroy( params );

    size_t patchCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( patch ) );
    TEST_ASSERT( patchCount == 1887 );

    MR_std_vector_MR_EdgeId* newBdEdges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( newBdEdges ) == 1 );

    MR_FaceBitSet_Destroy( patch );
    MR_std_vector_MR_EdgeId_Destroy( oldBdEdges );
    MR_std_vector_MR_EdgeId_Destroy( newBdEdges );
    MR_Mesh_Destroy( mesh );
}

void testRightBoundary( void )
{
    MR_Mesh* mesh = createMeshWithHoles();
    MR_std_vector_std_vector_MR_EdgeId* loops = MR_findRightBoundary_const_MR_FaceBitSet_ptr( MR_Mesh_Get_topology( mesh ), NULL );

    TEST_ASSERT( MR_std_vector_std_vector_MR_EdgeId_Size( loops ) == 2 );
    const MR_std_vector_MR_EdgeId* loop = MR_std_vector_std_vector_MR_EdgeId_At( loops, 0 );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( loop ) == 3 );

    loop = MR_std_vector_std_vector_MR_EdgeId_At( loops, 1 );
    TEST_ASSERT( MR_std_vector_MR_EdgeId_Size( loop ) == 4 );

    MR_std_vector_std_vector_MR_EdgeId_Destroy( loops );
    MR_Mesh_Destroy( mesh );
}

void testFindHoleComplicatingFaces( void )
{
    MR_Mesh* mesh = createMeshWithHoles();
    MR_FaceBitSet* faces = MR_findHoleComplicatingFaces( mesh );
    const size_t facesCount = MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( faces ) );
    TEST_ASSERT( facesCount == 0 );
    MR_FaceBitSet_Destroy( faces );
    MR_Mesh_Destroy( mesh );
}
