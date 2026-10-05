#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRBooleanOperation.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshCollidePrecise.h>
#include <MRMesh/MRIntersectionContour.h>
#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRConstants.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <tuple>

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


static Mesh makeBox( const Vector3f& min, const Vector3f& max )
{
    Box3f box;
    box.include( min );
    box.include( max );
    return makeCube( box.size(), box.min );
}

// Union of two not intersecting meshes must keep both of them,
// whatever the order of the arguments and the transformation of the second mesh are
static void expectUnionKeepsBoth( const Mesh& meshA, const Mesh& meshB )
{
    const double expected = meshA.volume() + meshB.volume();
    const auto xf = AffineXf3f::translation( { 7.3f, -2.1f, 0.6f } ) *
        AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1.f, 2.f, 3.f ).normalized(), 0.7f ) );

    for ( bool swapped : { false, true } )
    {
        const Mesh& m0 = swapped ? meshB : meshA;
        const Mesh& m1 = swapped ? meshA : meshB;

        const auto res = boolean( m0, m1, BooleanOperation::Union );
        ASSERT_TRUE( res.valid() );
        EXPECT_NEAR( res.mesh.volume(), expected, 1e-3 * expected );

        Mesh m1xf = m1;
        m1xf.transform( xf.inverse() );
        const auto resXf = boolean( m0, m1xf, BooleanOperation::Union, &xf );
        ASSERT_TRUE( resXf.valid() );
        EXPECT_NEAR( resXf.mesh.volume(), expected, 1e-3 * expected );
    }
}

// the boxes touch one another by the plane z = -41.50188 and no triangle of one crosses another,
// so the distance from any face of meshA there is zero and its sign is defined by rounding errors only
TEST( MRMesh, BooleanTouchingMeshes )
{
    expectUnionKeepsBoth(
        makeBox( { 11.3763f, -0.418446f, -41.50188f }, { 19.5763f, 7.61f, -36.20188f } ),
        makeBox( { 10.0763f, -3.418446f, -47.60188f }, { 21.0763f, 9.61f, -41.50188f } ) );
}

// closed wedge with a sharp convex edge along Y in the origin, opening towards -X
static Mesh makeWedge( float length, float halfWidth, float halfAngle )
{
    const float h = length * std::tan( halfAngle );
    Mesh res;
    res.points = std::vector<Vector3f>{
        { 0.f, -halfWidth, 0.f }, { 0.f, halfWidth, 0.f },
        { -length, -halfWidth, h }, { -length, halfWidth, h },
        { -length, -halfWidth, -h }, { -length, halfWidth, -h } };
    const Triangulation t = {
        { 0_v, 1_v, 3_v }, { 0_v, 3_v, 2_v },
        { 0_v, 4_v, 5_v }, { 0_v, 5_v, 1_v },
        { 2_v, 3_v, 5_v }, { 2_v, 5_v, 4_v },
        { 0_v, 2_v, 4_v }, { 1_v, 5_v, 3_v } };
    res.topology = MeshBuilder::fromTriangles( t );
    return res;
}

// the boxes are beyond the sharp edge of the wedge, where the planes of the two faces of that edge
// are on the opposite sides of them, and only the convexity of the edge tells inside from outside
TEST( MRMesh, BooleanBeyondSharpEdge )
{
    const Mesh wedge = makeWedge( 10.f, 5.f, 10.f * PI_F / 180.f );
    ASSERT_EQ( wedge.topology.findNumHoles(), 0 );

    for ( float angle : { 0.f, 1.05f, 1.31f, -1.05f, -1.31f } )
    {
        const Vector3f c = 2.f * Vector3f( std::cos( angle ), 0.f, std::sin( angle ) );
        expectUnionKeepsBoth( makeBox( c - Vector3f::diagonal( 0.05f ), c + Vector3f::diagonal( 0.05f ) ), wedge );
    }
}

TEST( MRMesh, BooleanDisjointMeshes )
{
    expectUnionKeepsBoth(
        makeBox( { 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f } ),
        makeBox( { 5.f, 5.f, 5.f }, { 6.f, 6.f, 6.f } ) );
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
    EXPECT_EQ( mapsA.cut2newFaces.size(), 280 );
    EXPECT_EQ( mapsA.cut2origin.size(), 280 );

    const auto& mapsB = mapper.getMaps( BooleanResultMapper::MapObject::B );
    EXPECT_EQ( mapsB.old2newVerts.size(), 160 );
    EXPECT_EQ( mapsB.cut2newFaces.size(), 320 );
    EXPECT_EQ( mapsB.cut2origin.size(), 320 );
}

// the spheres of the Boolean benchmark (B rotated around Z by n times 0.1 degree) have lone contours (each inside one triangle) for some n;
// subdivideLoneContours splits those triangles and updates the AABB tree instead of its rebuild
TEST( MRMesh, SubdivideLoneContoursUpdatesAABBTree )
{
    const Mesh sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    auto less = []( const VarEdgeTri & a, const VarEdgeTri & b )
    {
        return std::make_tuple( a.isEdgeATriB(), int( a.edge ), int( a.tri() ) ) < std::make_tuple( b.isEdgeATriB(), int( b.edge ), int( b.tri() ) );
    };
    for ( int n : { 50, 74 } )
    {
        const auto xf = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), float( double( n ) * 0.1f * PI / 180.0 ) ) );
        Mesh meshA = sphere;
        const Mesh & meshB = sphere;
        const auto conv = getVectorConverters( meshA, meshB, &xf );
        const auto contours = orderIntersectionContours( meshA.topology, meshB.topology,
            findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt, &xf ) );
        ContinuousContours loneA;
        for ( int i : detectLoneContours( contours ) )
            if ( !contours[i][0].isEdgeATriB() )
                loneA.push_back( contours[i] );
        OneMeshContours loneIntsA, loneIntsAonB;
        getOneMeshIntersectionContours( meshA, meshB, loneA, &loneIntsA, &loneIntsAonB, conv, &xf );
        removeLoneDegeneratedContours( meshB.topology, loneIntsA, loneIntsAonB );
        ASSERT_FALSE( loneIntsA.empty() );

        FaceBitSet loneFaces;
        for ( const auto & c : loneIntsA )
            loneFaces.autoResizeSet( std::get<FaceId>( c.intersections.front().primitiveId ) );
        const auto numFaces = meshA.topology.numValidFaces();
        subdivideLoneContours( meshA, loneIntsA );
        EXPECT_EQ( meshA.topology.numValidFaces(), numFaces + 2 * int( loneFaces.count() ) );
        const auto * tree = meshA.getAABBTreeNotCreate();
        ASSERT_TRUE( tree );
        EXPECT_EQ( tree->numLeaves(), size_t( meshA.topology.numValidFaces() ) );

        // same intersections as with a new tree
        Mesh rebuiltA = meshA;
        rebuiltA.invalidateCaches();
        auto updatedRes = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt, &xf );
        auto rebuiltRes = findCollidingEdgeTrisPrecise( rebuiltA, meshB, conv.toInt, &xf );
        std::sort( updatedRes.begin(), updatedRes.end(), less );
        std::sort( rebuiltRes.begin(), rebuiltRes.end(), less );
        EXPECT_EQ( updatedRes, rebuiltRes );

        EXPECT_TRUE( boolean( sphere, sphere, BooleanOperation::DifferenceAB, &xf ).valid() );
    }
}

// after splits of faces and edges in both meshes, updateCollidingEdgeTrisPrecise finds the same intersections as new search
TEST( MRMesh, UpdateCollidingEdgeTrisPrecise )
{
    Mesh meshA = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    Mesh meshB = meshA;
    const auto xf = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), 0.1f ) );
    const auto conv = getVectorConverters( meshA, meshB, &xf );
    auto res = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt, &xf );
    ASSERT_GT( res.size(), 100 );
    auto less = []( const VarEdgeTri & a, const VarEdgeTri & b )
    {
        return std::make_tuple( a.isEdgeATriB(), int( a.edge ), int( a.tri() ) ) < std::make_tuple( b.isEdgeATriB(), int( b.edge ), int( b.tri() ) );
    };
    auto orgRes = res;
    std::sort( orgRes.begin(), orgRes.end(), less );

    // split some intersected triangles of both meshes with new vertices inside the meshes, and an intersecting edge of A
    const VertId aFirstNewVert( meshA.topology.vertSize() );
    const VertId bFirstNewVert( meshB.topology.vertSize() );
    FaceHashMap aNew2Old, bNew2Old;
    for ( size_t i = 0; i < res.size(); i += 10 )
    {
        auto & mesh = res[i].isEdgeATriB() ? meshB : meshA;
        const auto f = res[i].tri();
        mesh.splitFace( f, mesh.triCenter( f ) - 0.01f * mesh.normal( f ), nullptr, res[i].isEdgeATriB() ? &bNew2Old : &aNew2Old );
    }
    const auto aEdgeIt = std::find_if( res.begin(), res.end(), []( const VarEdgeTri & et ) { return et.isEdgeATriB(); } );
    ASSERT_NE( aEdgeIt, res.end() );
    meshA.splitEdge( aEdgeIt->edge, nullptr, &aNew2Old );
    meshA.updateCachesAfterSplits( aNew2Old );
    meshB.updateCachesAfterSplits( bNew2Old );

    updateCollidingEdgeTrisPrecise( res, meshA, aFirstNewVert, meshB, bFirstNewVert, conv.toInt, &xf );
    auto newRes = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt, &xf );
    std::sort( res.begin(), res.end(), less );
    std::sort( newRes.begin(), newRes.end(), less );
    EXPECT_EQ( res, newRes );
    EXPECT_NE( res, orgRes );

    // no new vertices
    updateCollidingEdgeTrisPrecise( res, meshA, VertId( meshA.topology.vertSize() ), meshB, VertId( meshB.topology.vertSize() ), conv.toInt, &xf );
    EXPECT_EQ( res, newRes );
}

} //namespace MR
