#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRBooleanOperation.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRRegionBoundary.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, MeshBoolean )
{
    Mesh meshA = makeTorus( 1.1f, 0.5f, 8, 8 );
    Mesh meshB = makeTorus( 1.0f, 0.2f, 8, 8 );
    meshB.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), Vector3f::plusY() ) ) );

    const float shiftStep = 0.2f;
    const float angleStep = PI_F;/* *1.0f / 3.0f*/;
    const std::array<Vector3f, 3> baseAxis{Vector3f::plusX(),Vector3f::plusY(),Vector3f::plusZ()};
    for ( int maskTrans = 0; maskTrans < 8; ++maskTrans )
    {
        for ( int maskRot = 0; maskRot < 8; ++maskRot )
        {
            for ( float shift = 0.01f; shift < 0.2f; shift += shiftStep )
            {
                Vector3f shiftVec;
                for ( int i = 0; i < 3; ++i )
                    if ( maskTrans & ( 1 << i ) )
                        shiftVec += shift * baseAxis[i];
                for ( float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep )
                {
                    Matrix3f rotation;
                    for ( int i = 0; i < 3; ++i )
                        if ( maskRot & ( 1 << i ) )
                            rotation = Matrix3f::rotation( baseAxis[i], angle ) * rotation;

                    AffineXf3f xf;
                    xf = AffineXf3f::translation( shiftVec ) * AffineXf3f::linear( rotation );

                    EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation::Union, &xf ).valid() );
                    EXPECT_TRUE( boolean( meshB, meshA, BooleanOperation::Intersection, &xf ).valid() );
                }
            }
        }
    }
}


TEST( MRMesh, BooleanMultipleEdgePropogationSort )
{
    Mesh meshA;
    meshA.points = std::vector<Vector3f>
    {
        {0.0f,0.0f,0.0f},
        {-0.5f,1.0f,0.0f},
        {0.5f,1.0f,0.0f},
        {0.0f,1.5f,0.5f},
        {-1.0f,1.5f,0.0f},
        {1.0f,1.5f,0.0f}
    };
    Triangulation tA =
    {
        { 0_v, 2_v, 1_v },
        { 1_v, 2_v, 3_v },
        { 3_v, 4_v, 1_v },
        { 2_v, 5_v, 3_v },
        { 3_v, 5_v, 4_v }
    };
    meshA.topology = MeshBuilder::fromTriangles( tA );
    {
        Mesh meshASup = meshA;
        meshASup.points[3_v] = { 0.0f,1.5f,-0.5f };


        auto border = trackRightBoundaryLoop( meshA.topology, meshA.topology.findHoleRepresentiveEdges()[0] );

        meshA.addMeshPart( meshASup, true, { border }, { border } );
    }

    auto meshB = makeCube( Vector3f::diagonal( 2.0f ) );
    meshB.transform( AffineXf3f::translation( Vector3f( -1.5f, -0.2f, -0.5f ) ) );


    for ( int i = 0; i<int( BooleanOperation::Count ); ++i )
    {
        EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation( i ) ).valid() );
        EXPECT_TRUE( boolean( meshB, meshA, BooleanOperation( i ) ).valid() );
    }
}

TEST( MRMesh, BooleanResultMapper )
{
    Mesh meshA = makeTorus( 1.1f, 0.5f, 8, 8 );
    Mesh meshB = makeTorus( 1.0f, 0.2f, 8, 8 );
    meshB.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), Vector3f::plusY() ) ) );

    BooleanResultMapper mapper;
    BooleanParameters params;
    params.mapper = &mapper;

    const auto result = boolean( meshA, meshB, BooleanOperation::Union, params );
    EXPECT_TRUE( result.valid() );

    const auto& meshAValidVerts = meshA.topology.getValidVerts();
    const auto& meshBValidVerts = meshB.topology.getValidVerts();
    const auto vMapA = mapper.map( meshAValidVerts, BooleanResultMapper::MapObject::A );
    const auto vMapB = mapper.map( meshBValidVerts, BooleanResultMapper::MapObject::B );
    EXPECT_FALSE( vMapA.intersects( vMapB ) );
    EXPECT_EQ( vMapA.count(), 60 );
    EXPECT_EQ( vMapB.count(), 48 );

    const auto& meshAValidFaces = meshA.topology.getValidFaces();
    const auto& meshBValidFaces = meshB.topology.getValidFaces();
    const auto fMapA = mapper.map( meshAValidFaces, BooleanResultMapper::MapObject::A );
    const auto fMapB = mapper.map( meshBValidFaces, BooleanResultMapper::MapObject::B );
    EXPECT_FALSE( fMapA.intersects( fMapB ) );
    EXPECT_EQ( fMapA.count(), 224 );
    EXPECT_EQ( fMapB.count(), 192 );

    const auto newFaces = mapper.newFaces();
    EXPECT_EQ( newFaces.size(), 416 );
    EXPECT_EQ( newFaces.count(), 252 );

    const auto& mapsA = mapper.getMaps( BooleanResultMapper::MapObject::A );
    EXPECT_EQ( mapsA.old2newVerts.size(), 160 );
    EXPECT_EQ( mapsA.cut2newFaces.size(), 348 );
    EXPECT_EQ( mapsA.cut2origin.size(), 348 );

    const auto& mapsB = mapper.getMaps( BooleanResultMapper::MapObject::B );
    EXPECT_EQ( mapsB.old2newVerts.size(), 160 );
    EXPECT_EQ( mapsB.cut2newFaces.size(), 384 );
    EXPECT_EQ( mapsB.cut2origin.size(), 384 );
}

} //namespace MR
