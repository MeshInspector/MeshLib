#include "TestMacros.h"

#include "MRMeshBoolean.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshBoolean.h>
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRRegionBoundary.h>
#include <MRMeshC/MRTorus.h>
#include <MRMeshC/MRString.h>
#include <MRMeshC/MRVector3.h>

#define PI_F 3.14159265358979f
#define V( ID ) (MRVertId){ .id = ( ID ) }

void testMeshBoolean( void )
{
    MRMakeTorusParameters paramsA = { 1.1f, 0.5f, 8, 8 };
    MRMesh* meshA = mrMakeTorus( &paramsA );

    MRMakeTorusParameters paramsB = { 1.0f, 0.2f, 8, 8 };
    MRMesh* meshB = mrMakeTorus( &paramsB );
    {
        const MRVector3f plusZ = mrVector3fPlusZ();
        const MRVector3f plusY = mrVector3fPlusY();
        const MRMatrix3f rot = mrMatrix3fRotationVector( &plusZ, &plusY );
        const MRAffineXf3f xf = mrAffineXf3fLinear( &rot );
        mrMeshTransform( meshB, &xf, NULL );
    }

    const float shiftStep = 0.2f;
    const float angleStep = PI_F;/* *1.0f / 3.0f*/;
    const MRVector3f baseAxis[3] = {
        mrVector3fPlusX(),
        mrVector3fPlusY(),
        mrVector3fPlusZ()
    };
    for ( int maskTrans = 0; maskTrans < 8; ++maskTrans )
    {
        for ( int maskRot = 0; maskRot < 8; ++maskRot )
        {
            for ( float shift = 0.01f; shift < 0.2f; shift += shiftStep )
            {
                MRVector3f shiftVec = { 0.f, 0.f, 0.f };
                for ( int i = 0; i < 3; ++i )
                    if ( maskTrans & ( 1 << i ) )
                        shiftVec = mrVector3fAdd( &shiftVec, &baseAxis[i] );
                for ( float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep )
                {
                    MRMatrix3f rotation = mrMatrix3fIdentity();
                    for ( int i = 0; i < 3; ++i )
                    {
                        if ( maskRot & ( 1 << i ) )
                        {
                            const MRMatrix3f rot = mrMatrix3fRotationScalar( &baseAxis[i], angle );
                            rotation = mrMatrix3fMul( &rot, &rotation );
                        }
                    }

                    const MRAffineXf3f xf1 = mrAffineXf3fTranslation( &shiftVec );
                    const MRAffineXf3f xf2 = mrAffineXf3fLinear( &rotation );
                    const MRAffineXf3f xf = mrAffineXf3fMul( &xf1, &xf2 );

                    const MRBooleanParameters params = {
                        .rigidB2A = &xf,
                    };
                    MRBooleanResult resultAB = mrBoolean( meshA, meshB, MRBooleanOperationUnion, &params );
                    MRBooleanResult resultBA = mrBoolean( meshB, meshA, MRBooleanOperationUnion, &params );
                    TEST_ASSERT( mrStringSize( resultAB.errorString ) == 0 )
                    TEST_ASSERT( mrStringSize( resultBA.errorString ) == 0 )

                    mrStringFree( resultBA.errorString );
                    mrMeshFree( resultBA.mesh );
                    mrStringFree( resultAB.errorString );
                    mrMeshFree( resultAB.mesh );
                }
            }
        }
    }

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}

void testBooleanMultipleEdgePropogationSort( void )
{
    const MRVector3f points[6] = {
        {  0.0f, 0.0f, 0.0f },
        { -0.5f, 1.0f, 0.0f },
        { +0.5f, 1.0f, 0.0f },
        {  0.0f, 1.5f, 0.5f },
        { -1.0f, 1.5f, 0.0f },
        { +1.0f, 1.5f, 0.0f }
    };
    const MRThreeVertIds tA[5] = {
        { V( 0 ), V( 2 ), V( 1 ) },
        { V( 1 ), V( 2 ), V( 3 ) },
        { V( 3 ), V( 4 ), V( 1 ) },
        { V( 2 ), V( 5 ), V( 3 ) },
        { V( 3 ), V( 5 ), V( 4 ) }
    };
    MRMesh* meshA = mrMeshFromTriangles( points, 6, tA, 5 );

    {
        MRMesh* meshASup = mrMeshCopy( meshA );
        MRVector3f* meshASupPoints = mrMeshPointsRef( meshASup );
        meshASupPoints[3] = (MRVector3f){ 0.0f, 1.5f, -0.5f };

        const MRMeshTopology* meshATopology = mrMeshTopology( meshA );
        MREdgePath* meshAHoles = mrMeshTopologyFindHoleRepresentiveEdges( meshATopology );
        MREdgeLoop* border = mrTrackRightBoundaryLoop( meshATopology, meshAHoles->data[0], NULL );

        const MRFaceBitSet* meshASupFaces = mrMeshTopologyGetValidFaces( mrMeshTopology( meshASup ) );
        const MRMeshAddPartByMaskParameters params = {
            .flipOrientation = true,
            .thisContours = border,
            .thisContoursNum = 1,
            .fromContours = border,
            .fromContoursNum = 1,
        };
        mrMeshAddPartByMask( meshA, meshASup, meshASupFaces, &params );

        mrEdgePathFree( border );
        mrEdgePathFree( meshAHoles );
        mrMeshFree( meshASup );
    }

    const MRVector3f meshBSize = mrVector3fDiagonal( 2.0f );
    const MRVector3f meshBBase = mrVector3fDiagonal( -0.5f );
    MRMesh* meshB = mrMakeCube( &meshBSize, &meshBBase );
    {
        const MRVector3f v1 = { -1.5f, -0.2f, -0.5f };
        const MRAffineXf3f xf1 = mrAffineXf3fTranslation( &v1 );
        mrMeshTransform( meshB, &xf1, NULL );
    }

    for ( int i = 0; i < MRBooleanOperationCount; ++i )
    {
        MRBooleanResult resultAB = mrBoolean( meshA, meshB, (MRBooleanOperation)i, NULL );
        MRBooleanResult resultBA = mrBoolean( meshB, meshA, (MRBooleanOperation)i, NULL );
        TEST_ASSERT( mrStringSize( resultAB.errorString ) == 0 )
        TEST_ASSERT( mrStringSize( resultBA.errorString ) == 0 )

        mrStringFree( resultBA.errorString );
        mrMeshFree( resultBA.mesh );
        mrStringFree( resultAB.errorString );
        mrMeshFree( resultAB.mesh );
    }

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}

void testBooleanMapper( void )
{
    MRMakeTorusParameters paramsA = { 1.1f, 0.5f, 8, 8 };
    MRMesh* meshA = mrMakeTorus( &paramsA );

    MRMakeTorusParameters paramsB = { 1.0f, 0.2f, 8, 8 };
    MRMesh* meshB = mrMakeTorus( &paramsB );
    {
        const MRVector3f plusZ = mrVector3fPlusZ();
        const MRVector3f plusY = mrVector3fPlusY();
        const MRMatrix3f rot = mrMatrix3fRotationVector( &plusZ, &plusY );
        const MRAffineXf3f xf = mrAffineXf3fLinear( &rot );
        mrMeshTransform( meshB, &xf, NULL );
    }

    MRBooleanResultMapper* mapper = mrBooleanResultMapperNew();
    MRBooleanParameters parameters = {
        .mapper = mapper,
    };
    MRBooleanResult result = mrBoolean( meshA, meshB, MRBooleanOperationUnion, &parameters );
    TEST_ASSERT( mrStringSize( result.errorString ) == 0 )

    const MRVertBitSet* meshAValidPoints = mrMeshTopologyGetValidVerts( mrMeshTopology( meshA ) );
    const MRVertBitSet* meshBValidPoints = mrMeshTopologyGetValidVerts( mrMeshTopology( meshB ) );
    MRVertBitSet* vMapA = mrBooleanResultMapperMapVerts( mapper, meshAValidPoints, MRBooleanResultMapperMapObjectA );
    MRVertBitSet* vMapB = mrBooleanResultMapperMapVerts( mapper, meshBValidPoints, MRBooleanResultMapperMapObjectB );
    TEST_ASSERT( mrBitSetSize( (MRBitSet*)vMapA ) == 60 )
    TEST_ASSERT( mrBitSetSize( (MRBitSet*)vMapB ) == 204 )
    TEST_ASSERT( mrBitSetCount( (MRBitSet*)vMapA ) == 60 )
    TEST_ASSERT( mrBitSetCount( (MRBitSet*)vMapB ) == 48 )

    const MRFaceBitSet* meshAValidFaces = mrMeshTopologyGetValidFaces( mrMeshTopology( meshA ) );
    const MRFaceBitSet* meshBValidFaces = mrMeshTopologyGetValidFaces( mrMeshTopology( meshB ) );
    MRFaceBitSet* fMapA = mrBooleanResultMapperMapFaces( mapper, meshAValidFaces, MRBooleanResultMapperMapObjectA );
    MRFaceBitSet* fMapB = mrBooleanResultMapperMapFaces( mapper, meshBValidFaces, MRBooleanResultMapperMapObjectB );
    TEST_ASSERT( mrBitSetSize( (MRBitSet*)fMapA ) == 224 )
    TEST_ASSERT( mrBitSetSize( (MRBitSet*)fMapB ) == 416 )
    TEST_ASSERT( mrBitSetCount( (MRBitSet*)fMapA ) == 224 )
    TEST_ASSERT( mrBitSetCount( (MRBitSet*)fMapB ) == 192 )

    MRFaceBitSet* newFaces = mrBooleanResultMapperNewFaces( mapper );
    TEST_ASSERT( mrBitSetSize( (MRBitSet*)newFaces ) == 416 )
    TEST_ASSERT( mrBitSetCount( (MRBitSet*)newFaces ) == 252 )

    const MRBooleanResultMapperMaps* mapsA = mrBooleanResultMapperGetMaps( mapper, MRBooleanResultMapperMapObjectA );
    TEST_ASSERT( !mrBooleanResultMapperMapsIdentity( mapsA ) )
    TEST_ASSERT( mrBooleanResultMapperMapsOld2NewVerts( mapsA ).size == 160 )
    TEST_ASSERT( mrBooleanResultMapperMapsCut2newFaces( mapsA ).size == 348 )
    TEST_ASSERT( mrBooleanResultMapperMapsCut2origin( mapsA ).size == 348 )

    const MRBooleanResultMapperMaps* mapsB = mrBooleanResultMapperGetMaps( mapper, MRBooleanResultMapperMapObjectB );
    TEST_ASSERT( !mrBooleanResultMapperMapsIdentity( mapsB ) )
    TEST_ASSERT( mrBooleanResultMapperMapsOld2NewVerts( mapsB ).size == 160 )
    TEST_ASSERT( mrBooleanResultMapperMapsCut2newFaces( mapsB ).size == 384 )
    TEST_ASSERT( mrBooleanResultMapperMapsCut2origin( mapsB ).size == 384 )

    mrFaceBitSetFree( newFaces );

    mrFaceBitSetFree( fMapB );
    mrFaceBitSetFree( fMapA );

    mrVertBitSetFree( vMapB );
    mrVertBitSetFree( vMapA );

    mrStringFree( result.errorString );
    mrMeshFree( result.mesh );

    mrBooleanResultMapperFree( mapper );

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}
