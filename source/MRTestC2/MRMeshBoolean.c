#include "TestMacros.h"

#include "MRMeshBoolean.h"

#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRId.h>
#include <MRCMesh/MRMatrix3.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBoolean.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRRegionBoundary.h>
#include <MRCMesh/MRString.h>
#include <MRCMesh/MRTorus.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/std_array_MR_VertId_3.h>
#include <MRCMisc/std_vector_MR_Vector3f.h>
#include <MRCMisc/std_vector_MR_EdgeId.h>
#include <MRCMisc/std_vector_std_vector_MR_EdgeId.h>

#define PI_F 3.14159265358979f
#define V( ID ) (MRVertId){ .id = ( ID ) }

void testMeshBoolean( void )
{
    float primaryRadiusA = 1.1f;
    float secondaryRadiusA = 0.5f;
    int32_t primaryResolutionA = 8;
    int32_t secondaryResolutionA = 8;
    MR_Mesh* meshA = MR_makeTorus( &primaryRadiusA, &secondaryRadiusA, &primaryResolutionA, &secondaryResolutionA, NULL );

    float primaryRadiusB = 1.0f;
    float secondaryRadiusB = 0.2f;
    int32_t primaryResolutionB = 8;
    int32_t secondaryResolutionB = 8;
    MR_Mesh* meshB = MR_makeTorus( &primaryRadiusB, &secondaryRadiusB, &primaryResolutionB, &secondaryResolutionB, NULL );

    {
        const MR_Vector3f plusZ = MR_Vector3f_plusZ();
        const MR_Vector3f plusY = MR_Vector3f_plusY();
        const MR_Matrix3f rot = MR_Matrix3f_rotation_MR_Vector3f( &plusZ, &plusY );
        const MR_AffineXf3f xf = MR_AffineXf3f_linear( &rot );
        MR_Mesh_transform( meshB, &xf, NULL );
    }

    const float shiftStep = 0.2f;
    const float angleStep = PI_F;/* *1.0f / 3.0f*/;
    const MR_Vector3f baseAxis[3] = {
        MR_Vector3f_plusX(),
        MR_Vector3f_plusY(),
        MR_Vector3f_plusZ()
    };
    for ( int maskTrans = 0; maskTrans < 8; ++maskTrans )
    {
        for ( int maskRot = 0; maskRot < 8; ++maskRot )
        {
            for ( float shift = 0.01f; shift < 0.2f; shift += shiftStep )
            {
                MR_Vector3f shiftVec = { 0.f, 0.f, 0.f };
                for ( int i = 0; i < 3; ++i )
                    if ( maskTrans & ( 1 << i ) )
                        shiftVec = MR_add_MR_Vector3f( &shiftVec, &baseAxis[i] );
                for ( float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep )
                {
                    MR_Matrix3f rotation = MR_Matrix3f_identity();
                    for ( int i = 0; i < 3; ++i )
                    {
                        if ( maskRot & ( 1 << i ) )
                        {
                            const MR_Matrix3f rot = MR_Matrix3f_rotation_float( &baseAxis[i], angle );
                            rotation = MR_mul_MR_Matrix3f( &rot, &rotation );
                        }
                    }

                    const MR_AffineXf3f xf1 = MR_AffineXf3f_translation( &shiftVec );
                    const MR_AffineXf3f xf2 = MR_AffineXf3f_linear( &rotation );
                    const MR_AffineXf3f xf = MR_mul_MR_AffineXf3f( &xf1, &xf2 );

                    MR_BooleanParameters *params = MR_BooleanParameters_DefaultConstruct();
                    MR_BooleanParameters_Set_rigidB2A( params, &xf );

                    MR_BooleanResult *resultAB = MR_boolean_4_const_MR_Mesh_ref( meshA, meshB, MR_BooleanOperation_Union, params );
                    MR_BooleanResult *resultBA = MR_boolean_4_const_MR_Mesh_ref( meshB, meshA, MR_BooleanOperation_Union, params );
                    TEST_ASSERT( MR_BooleanResult_valid( resultAB ) )
                    TEST_ASSERT( MR_BooleanResult_valid( resultBA ) )

                    MR_BooleanResult_Destroy( resultAB );
                    MR_BooleanResult_Destroy( resultBA );

                    MR_BooleanParameters_Destroy( params );
                }
            }
        }
    }

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}

void testBooleanMultipleEdgePropogationSort( void )
{
    MR_VertCoords* pointsVec = MR_VertCoords_Construct_1_uint64_t( 6 );
    *MR_VertCoords_index( pointsVec, (MR_VertId){0} ) = (MR_Vector3f){  0.0f, 0.0f, 0.0f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){1} ) = (MR_Vector3f){ -0.5f, 1.0f, 0.0f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){2} ) = (MR_Vector3f){ +0.5f, 1.0f, 0.0f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){3} ) = (MR_Vector3f){  0.0f, 1.5f, 0.5f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){4} ) = (MR_Vector3f){ -1.0f, 1.5f, 0.0f };
    *MR_VertCoords_index( pointsVec, (MR_VertId){5} ) = (MR_Vector3f){ +1.0f, 1.5f, 0.0f };

    MR_Triangulation *triangulation = MR_Triangulation_Construct_1_uint64_t( 5 );
    *MR_Triangulation_index( triangulation, (MR_FaceId){0} ) = (MR_std_array_MR_VertId_3){{ {0}, {2}, {1} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){1} ) = (MR_std_array_MR_VertId_3){{ {1}, {2}, {3} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){2} ) = (MR_std_array_MR_VertId_3){{ {3}, {4}, {1} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){3} ) = (MR_std_array_MR_VertId_3){{ {2}, {5}, {3} }};
    *MR_Triangulation_index( triangulation, (MR_FaceId){4} ) = (MR_std_array_MR_VertId_3){{ {3}, {5}, {4} }};

    MR_Mesh* meshA = MR_Mesh_fromTriangles( MR_PassBy_Move, pointsVec, triangulation, NULL, MR_PassBy_DefaultArgument, NULL );

    MR_VertCoords_Destroy( pointsVec );
    MR_Triangulation_Destroy( triangulation );

    {
        MR_Mesh* meshASup = MR_Mesh_ConstructFromAnother( MR_PassBy_Copy, meshA );
        MR_Vector3f* meshASupPoints = MR_VertCoords_data( MR_Mesh_GetMutable_points( meshASup ) );
        meshASupPoints[3] = (MR_Vector3f){ 0.0f, 1.5f, -0.5f };

        const MR_MeshTopology* meshATopology = MR_Mesh_Get_topology( meshA );
        MR_std_vector_MR_EdgeId* meshAHoles = MR_MeshTopology_findHoleRepresentiveEdges( meshATopology, NULL );

        MR_std_vector_MR_EdgeId* border = MR_trackRightBoundaryLoop_MR_EdgeId( meshATopology, *MR_std_vector_MR_EdgeId_At( meshAHoles, 0 ), NULL );

        MR_std_vector_std_vector_MR_EdgeId* borderVec = MR_std_vector_std_vector_MR_EdgeId_DefaultConstruct();
        MR_std_vector_std_vector_MR_EdgeId_PushBack( borderVec, MR_PassBy_Move, border );
        MR_std_vector_MR_EdgeId_Destroy( border );

        const MR_FaceBitSet* meshASupFaces = MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( meshASup ) );

        MR_MeshPart *mp = MR_MeshPart_Construct( meshASup, meshASupFaces );
        MR_Mesh_addMeshPart_5( meshA, mp, &(bool){true}, borderVec, borderVec, NULL );
        MR_MeshPart_Destroy( mp );

        MR_std_vector_std_vector_MR_EdgeId_Destroy( borderVec );
        MR_std_vector_MR_EdgeId_Destroy( meshAHoles );
        MR_Mesh_Destroy( meshASup );
    }

    const MR_Vector3f meshBSize = MR_Vector3f_diagonal( 2.0f );
    const MR_Vector3f meshBBase = MR_Vector3f_diagonal( -0.5f );
    MR_Mesh* meshB = MR_makeCube( &meshBSize, &meshBBase );
    {
        const MR_Vector3f v1 = { -1.5f, -0.2f, -0.5f };
        const MR_AffineXf3f xf1 = MR_AffineXf3f_translation( &v1 );
        MR_Mesh_transform( meshB, &xf1, NULL );
    }

    for ( int i = 0; i < MR_BooleanOperation_Count; ++i )
    {
        MR_BooleanResult *resultAB = MR_boolean_4_const_MR_Mesh_ref( meshA, meshB, i, NULL );
        MR_BooleanResult *resultBA = MR_boolean_4_const_MR_Mesh_ref( meshB, meshA, i, NULL );
        TEST_ASSERT( MR_BooleanResult_valid( resultAB ) )
        TEST_ASSERT( MR_BooleanResult_valid( resultBA ) )

        MR_BooleanResult_Destroy( resultAB );
        MR_BooleanResult_Destroy( resultBA );
    }

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}

void testBooleanMapper( void )
{
    float primaryRadiusA = 1.1f;
    float secondaryRadiusA = 0.5f;
    int32_t primaryResolutionA = 8;
    int32_t secondaryResolutionA = 8;
    MR_Mesh* meshA = MR_makeTorus(&primaryRadiusA, &secondaryRadiusA, &primaryResolutionA, &secondaryResolutionA, NULL);

    float primaryRadiusB = 1.0f;
    float secondaryRadiusB = 0.2f;
    int32_t primaryResolutionB = 8;
    int32_t secondaryResolutionB = 8;
    MR_Mesh* meshB = MR_makeTorus(&primaryRadiusB, &secondaryRadiusB, &primaryResolutionB, &secondaryResolutionB, NULL);

    {
        const MR_Vector3f plusZ = MR_Vector3f_plusZ();
        const MR_Vector3f plusY = MR_Vector3f_plusY();
        const MR_Matrix3f rot = MR_Matrix3f_rotation_MR_Vector3f( &plusZ, &plusY );
        const MR_AffineXf3f xf = MR_AffineXf3f_linear( &rot );
        MR_Mesh_transform( meshB, &xf, NULL );
    }

    MR_BooleanResultMapper* mapper = MR_BooleanResultMapper_DefaultConstruct();
    MR_BooleanParameters* parameters = MR_BooleanParameters_DefaultConstruct();
    MR_BooleanParameters_Set_mapper( parameters, mapper );

    MR_BooleanResult* result = MR_boolean_4_const_MR_Mesh_ref( meshA, meshB, MR_BooleanOperation_Union, parameters );
    MR_BooleanParameters_Destroy( parameters );

    TEST_ASSERT( MR_BooleanResult_valid( result ) )

    const MR_VertBitSet* meshAValidPoints = MR_MeshTopology_getValidVerts( MR_Mesh_Get_topology( meshA ) );
    const MR_VertBitSet* meshBValidPoints = MR_MeshTopology_getValidVerts( MR_Mesh_Get_topology( meshB ) );
    MR_VertBitSet* vMapA = MR_BooleanResultMapper_map_MR_VertBitSet( mapper, meshAValidPoints, MR_BooleanResultMapper_MapObject_A );
    MR_VertBitSet* vMapB = MR_BooleanResultMapper_map_MR_VertBitSet( mapper, meshBValidPoints, MR_BooleanResultMapper_MapObject_B );
    TEST_ASSERT( MR_BitSet_size( MR_VertBitSet_UpcastTo_MR_BitSet( vMapA ) ) == 60 )
    TEST_ASSERT( MR_BitSet_size( MR_VertBitSet_UpcastTo_MR_BitSet( vMapB ) ) == 204 )
    TEST_ASSERT( MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( vMapA ) ) == 60 )
    TEST_ASSERT( MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( vMapB ) ) == 48 )

    const MR_FaceBitSet* meshAValidFaces = MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( meshA ) );
    const MR_FaceBitSet* meshBValidFaces = MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( meshB ) );
    MR_FaceBitSet* fMapA = MR_BooleanResultMapper_map_MR_FaceBitSet( mapper, meshAValidFaces, MR_BooleanResultMapper_MapObject_A );
    MR_FaceBitSet* fMapB = MR_BooleanResultMapper_map_MR_FaceBitSet( mapper, meshBValidFaces, MR_BooleanResultMapper_MapObject_B );
    TEST_ASSERT( MR_BitSet_size( MR_FaceBitSet_UpcastTo_MR_BitSet( fMapA ) ) == 224 )
    TEST_ASSERT( MR_BitSet_size( MR_FaceBitSet_UpcastTo_MR_BitSet( fMapB ) ) == 416 )
    TEST_ASSERT( MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( fMapA ) ) == 224 )
    TEST_ASSERT( MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( fMapB ) ) == 192 )

    MR_FaceBitSet* newFaces = MR_BooleanResultMapper_newFaces( mapper );
    TEST_ASSERT( MR_BitSet_size( MR_FaceBitSet_UpcastTo_MR_BitSet( newFaces ) ) == 416 )
    TEST_ASSERT( MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( newFaces ) ) == 252 )

    const MR_BooleanResultMapper_Maps* mapsA = MR_BooleanResultMapper_getMaps( mapper, MR_BooleanResultMapper_MapObject_A );
    TEST_ASSERT( !*MR_BooleanResultMapper_Maps_Get_identity( mapsA ) )
    TEST_ASSERT( MR_VertMap_size( MR_BooleanResultMapper_Maps_Get_old2newVerts( mapsA ) ) == 160 )
    TEST_ASSERT( MR_FaceMap_size( MR_BooleanResultMapper_Maps_Get_cut2newFaces( mapsA ) ) == 348 )
    TEST_ASSERT( MR_FaceMap_size( MR_BooleanResultMapper_Maps_Get_cut2origin( mapsA ) ) == 348 )

    const MR_BooleanResultMapper_Maps* mapsB = MR_BooleanResultMapper_getMaps( mapper, MR_BooleanResultMapper_MapObject_B );
    TEST_ASSERT( !*MR_BooleanResultMapper_Maps_Get_identity( mapsB ) )
    TEST_ASSERT( MR_VertMap_size( MR_BooleanResultMapper_Maps_Get_old2newVerts( mapsB ) ) == 160 )
    TEST_ASSERT( MR_FaceMap_size( MR_BooleanResultMapper_Maps_Get_cut2newFaces( mapsB ) ) == 384 )
    TEST_ASSERT( MR_FaceMap_size( MR_BooleanResultMapper_Maps_Get_cut2origin( mapsB ) ) == 384 )

    MR_FaceBitSet_Destroy( newFaces );

    MR_FaceBitSet_Destroy( fMapB );
    MR_FaceBitSet_Destroy( fMapA );

    MR_VertBitSet_Destroy( vMapB );
    MR_VertBitSet_Destroy( vMapA );

    MR_BooleanResult_Destroy( result );

    MR_BooleanResultMapper_Destroy( mapper );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}
