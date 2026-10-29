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
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRIntersectionContour.h>
#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRRegularGridMesh.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MREmbedTerrainStructure.h>
#include <MRMesh/MRMeshProject.h>
#include <MRPch/MRSuppressWarning.h>
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
    res.points = VertCoords{
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

// a small sphere pierces a big triangle of the cube without crossing its edges (a lone contour), alone, so that nothing else
// cuts the triangle, or with another sphere crossing the edges of the same triangle, or in the hole of a torus lying in the plane
// of the triangle, which cuts it along two nested circles around the sphere's one; the cut connects the lone contours with the rest
// of the triangle, and the mapper maps the faces made of the triangle back to it
TEST( MRMesh, BooleanLoneContours )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    Mesh lone = makeSphere( { .radius = 0.2f, .numMeshVertices = 200 } );
    lone.transform( AffineXf3f::translation( Vector3f( 1.f, -0.4f, 2.f ) ) );
    Mesh crossing = makeSphere( { .radius = 0.3f, .numMeshVertices = 300 } );
    crossing.transform( AffineXf3f::translation( Vector3f( 0.f, 0.f, 2.f ) ) );
    Mesh both = lone;
    both.addMesh( crossing );
    // the torus cuts the triangle along circles of radii about 0.4 and 0.8, and the small sphere in its hole along one of radius 0.15,
    // around a point 1.13 or more from the sides; the torus is a little above the plane z = 2, which its inner circle of vertices would be in otherwise
    Mesh islands = makeTorus( 0.6f, 0.2f, 16, 8 );
    islands.transform( AffineXf3f::translation( Vector3f( 0.8f, -0.8f, 2.01f ) ) );
    Mesh inner = makeSphere( { .radius = 0.15f, .numMeshVertices = 100 } );
    inner.transform( AffineXf3f::translation( Vector3f( 0.8f, -0.8f, 2.f ) ) );
    islands.addMesh( inner );
    const auto xf = AffineXf3f::translation( Vector3f( 0.3f, -0.2f, 0.25f ) ) *
        AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1.f, 2.f, 3.f ).normalized(), 0.7f ) );
    // the triangle of the lone contours on the top z = 2 of the cube, below its diagonal y = x
    const FaceId loneFace = findProjection( Vector3f( 1.f, -0.4f, 2.f ), cube ).proj.face;
    ASSERT_TRUE( loneFace.valid() );

    // every face of a result comes from a valid face of one input; the faces in the plane of the triangle strictly inside it come from it,
    // and those from it are inside it or on its sides
    auto checkMapper = [&] ( const BooleanResult& r, const BooleanResultMapper& mapper, const Mesh& a, const Mesh& b, bool cubeIsA )
    {
        const auto mapA = mapper.getNew2OldFaceMap( BooleanResultMapper::MapObject::A );
        const auto mapB = mapper.getNew2OldFaceMap( BooleanResultMapper::MapObject::B );
        const float eps = 1e-4f;
        int numInside = 0;
        for ( auto f : r.mesh.topology.getValidFaces() )
        {
            const auto fa = f < mapA.size() ? mapA[f] : FaceId();
            const auto fb = f < mapB.size() ? mapB[f] : FaceId();
            EXPECT_NE( fa.valid(), fb.valid() );
            EXPECT_TRUE( fa ? a.topology.hasFace( fa ) : b.topology.hasFace( fb ) );
            const bool fromLone = ( cubeIsA ? fa : fb ) == loneFace;
            const auto c = r.mesh.triCenter( f );
            // the slivers along the contours have no reliable normal, while the other faces of the spheres there are nearly vertical
            if ( std::abs( c.z - 2.f ) < eps && r.mesh.area( f ) > 1e-6 && std::abs( r.mesh.normal( f ).z ) > 0.99f
                && c.y < c.x - eps && c.x < 2.f - eps && c.y > -2.f + eps )
            {
                EXPECT_TRUE( fromLone );
                ++numInside;
            }
            if ( fromLone )
            {
                EXPECT_TRUE( std::abs( c.z - 2.f ) < eps && c.y < c.x + eps && c.x < 2.f + eps && c.y > -2.f - eps );
            }
        }
        EXPECT_GT( numInside, 0 );
    };

    for ( const Mesh* spheres : { &lone, &both, &islands } )
    {
        const double tol = 1e-5 * ( cube.volume() + spheres->volume() );
        for ( bool swap : { false, true } )
        {
            for ( const AffineXf3f* rigidB2A : { (const AffineXf3f*)nullptr, &xf } )
            {
                // rigidB2A moves the second mesh back in place
                const Mesh& a = swap ? *spheres : cube;
                Mesh b = swap ? cube : *spheres;
                if ( rigidB2A )
                    b.transform( rigidB2A->inverse() );
                BooleanResultMapper interMapper, uniMapper, diffMapper;
                const auto inter = boolean( a, b, BooleanOperation::Intersection, rigidB2A, &interMapper );
                const auto uni = boolean( a, b, BooleanOperation::Union, rigidB2A, &uniMapper );
                const auto diff = boolean( a, b, BooleanOperation::DifferenceAB, rigidB2A, &diffMapper );
                ASSERT_TRUE( inter.valid() ) << inter.errorString;
                ASSERT_TRUE( uni.valid() ) << uni.errorString;
                ASSERT_TRUE( diff.valid() ) << diff.errorString;
                for ( const auto* r : { &inter, &uni, &diff } )
                    EXPECT_EQ( r->mesh.topology.findNumHoles(), 0 );
                // an island left in its triangle would make a separate component; the components of the intersection are the lower halves
                // of the spheres and of the torus, which do not touch one another
                EXPECT_EQ( MeshComponents::getNumComponents( uni.mesh ), 1 );
                EXPECT_EQ( MeshComponents::getNumComponents( inter.mesh ), spheres == &lone ? 1 : 2 );
                EXPECT_NEAR( uni.mesh.volume() + inter.mesh.volume(), a.volume() + b.volume(), tol );
                EXPECT_NEAR( diff.mesh.volume() + inter.mesh.volume(), a.volume(), tol );
                checkMapper( inter, interMapper, a, b, !swap );
                checkMapper( uni, uniMapper, a, b, !swap );
                checkMapper( diff, diffMapper, a, b, !swap );
            }
        }
    }
}

// the area of the faces of a result in the top z = 1 of the cube [-1,1]^3, facing up
static double topArea( const Mesh& m )
{
    double res = 0;
    for ( auto f : m.topology.getValidFaces() )
        if ( std::abs( m.triCenter( f ).z - 1.f ) < 1e-5f && m.normal( f ).z > 0 )
            res += m.area( f );
    return res;
}

// two overlapping boxes (one mesh crossing itself) poke through one triangle of the top of the cube; the forced cut splits their contours
// where they cross, and the winding numbers of the boxes (0 outside both, 1 inside one, 2 inside both) tell the parts of the cube, so it keeps
// its top outside the boxes, or inside them for the intersection; the forced boolean is one closed shell
TEST( MRMesh, BooleanForceCutCrossingContours )
{
    const auto cube = makeCube( Vector3f::diagonal( 2.f ), Vector3f::diagonal( -1.f ) );
    auto boxes = makeCube( Vector3f( 0.2f, 0.2f, 1.f ), Vector3f( 0.5f, -0.1f, 0.5f ) );
    boxes.addMesh( makeCube( Vector3f( 0.2f, 0.2f, 1.1f ), Vector3f( 0.55f, -0.05f, 0.45f ) ) );
    // the union of the footprints of the boxes on the top z = 1 of the cube
    const double footprints = 2 * 0.04 - 0.15 * 0.15;
    for ( bool cubeIsA : { true, false } )
    {
        const auto& a = cubeIsA ? cube : boxes;
        const auto& b = cubeIsA ? boxes : cube;
        for ( auto op : { BooleanOperation::Union, BooleanOperation::Intersection, cubeIsA ? BooleanOperation::DifferenceAB : BooleanOperation::DifferenceBA } )
        {
            const auto res = boolean( a, b, op, { .forceCut = true } );
            ASSERT_TRUE( res.valid() ) << res.errorString;
            EXPECT_EQ( ( cubeIsA ? res.meshABadContourFaces : res.meshBBadContourFaces ).count(), 1 );
            EXPECT_NEAR( topArea( res.mesh ), op == BooleanOperation::Intersection ? footprints : 4 - footprints, 1e-5 );
            const auto force = forceBoolean( a, b, op );
            ASSERT_TRUE( force.valid() ) << force.errorString;
            EXPECT_TRUE( force.mesh.topology.isClosed() );
            EXPECT_EQ( MeshComponents::getNumComponents( force.mesh ), 1 );
        }
    }
}

// the forced boolean decides the parts of the cube by the exact winding numbers of mesh B, not counted from the lowest part:
// a box nested in another one (not crossing) pokes through the top of the cube, so the ring between their contours is to the left of one
// closed path and to the right of the other (0 outside, 1 in the ring, 2 inside the inner box); a box enclosing everything makes all of
// the cube inside, also where two boxes cross; and where an inside-out box crosses another box, only the top inside the first box and
// outside the second one is inside (the winding number is -1 inside the second box only); with mesh B in its own space too
TEST( MRMesh, BooleanForceCutExactWindings )
{
    const auto cube = makeBox( Vector3f::diagonal( -1.f ), Vector3f::diagonal( 1.f ) );
    auto nested = makeBox( Vector3f( 0.1f, -0.3f, 0.5f ), Vector3f( 0.7f, 0.3f, 1.5f ) );
    nested.addMesh( makeBox( Vector3f( 0.2f, -0.2f, 0.6f ), Vector3f( 0.6f, 0.2f, 1.4f ) ) );
    auto nestedEnclosed = nested;
    nestedEnclosed.addMesh( makeBox( Vector3f::diagonal( -2.f ), Vector3f::diagonal( 2.f ) ) );
    auto enclosed = makeBox( Vector3f( 0.5f, -0.1f, 0.5f ), Vector3f( 0.7f, 0.1f, 1.5f ) );
    enclosed.addMesh( makeBox( Vector3f( 0.55f, -0.05f, 0.45f ), Vector3f( 0.75f, 0.15f, 1.55f ) ) );
    enclosed.addMesh( makeBox( Vector3f::diagonal( -2.f ), Vector3f::diagonal( 2.f ) ) );
    auto inverted = makeBox( Vector3f( 0.3f, -0.6f, 0.5f ), Vector3f( 0.7f, -0.2f, 1.5f ) );
    auto insideOut = makeBox( Vector3f( 0.4f, -0.5f, 0.45f ), Vector3f( 0.8f, -0.1f, 1.55f ) );
    insideOut.topology.flipOrientation();
    inverted.addMesh( insideOut );
    const AffineXf3f b2a( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.7f ), Vector3f( 0.3f, -0.2f, 0.1f ) );
    for ( const auto& [b, insideTop] : { std::pair( &nested, 0.6 * 0.6 ), std::pair( &nestedEnclosed, 4. ), std::pair( &enclosed, 4. ),
        std::pair( &inverted, 0.4 * 0.4 - 0.3 * 0.3 ) } )
    {
        auto bInOwnSpace = *b;
        bInOwnSpace.transform( b2a.inverse() );
        for ( bool xf : { false, true } )
            for ( auto op : { BooleanOperation::Union, BooleanOperation::Intersection } )
            {
                const auto res = boolean( cube, xf ? bInOwnSpace : *b, op, { .rigidB2A = xf ? &b2a : nullptr, .forceCut = true } );
                ASSERT_TRUE( res.valid() ) << res.errorString;
                EXPECT_NEAR( topArea( res.mesh ), op == BooleanOperation::Intersection ? insideTop : 4 - insideTop, 1e-4 );
            }
    }
}

// the precut output of the mesh that the operation does not cut (A for InsideB, B for InsideA) takes it only after the cut
// of the other mesh, which sorts its intersections by that mesh itself, not by a copy; here the cut of the cube connects
// a lone contour, whose exact points are made of the vertices of the sphere
TEST( MRMesh, BooleanPreCutOfUncutMesh )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    Mesh sphere = makeSphere( { .radius = 0.2f, .numMeshVertices = 200 } );
    sphere.transform( AffineXf3f::translation( Vector3f( 1.f, -0.4f, 2.f ) ) );

    for ( bool swap : { false, true } )
    {
        const Mesh& a = swap ? sphere : cube;
        const Mesh& b = swap ? cube : sphere;
        const auto op = swap ? BooleanOperation::InsideB : BooleanOperation::InsideA;
        const auto cubeObj = swap ? BooleanResultMapper::MapObject::B : BooleanResultMapper::MapObject::A;
        BooleanResultMapper fullMapper;
        const auto full = boolean( a, b, op, { .mapper = &fullMapper } );
        ASSERT_TRUE( full.valid() ) << full.errorString;
        EXPECT_TRUE( full.mesh.topology.checkValidity() );
        EXPECT_EQ( full.mesh.topology.findNumHoles(), 1 ); // the disc of the cube inside the sphere
        EXPECT_GT( full.mesh.topology.numValidFaces(), 0 );

        // with the precut output, through the const& overload and the Mesh&& one, the cube is cut as without it
        for ( bool rvalue : { false, true } )
        {
            BooleanPreCutResult preCut;
            BooleanResultMapper mapper;
            const BooleanParameters params{ .mapper = &mapper, .outPreCutA = swap ? &preCut : nullptr, .outPreCutB = swap ? nullptr : &preCut };
            const auto res = rvalue ? boolean( Mesh( a ), Mesh( b ), op, params ) : boolean( a, b, op, params );
            ASSERT_TRUE( res.valid() ) << res.errorString;
            EXPECT_TRUE( res.meshABadContourFaces.none() && res.meshBBadContourFaces.none() );
            EXPECT_TRUE( preCut.mesh == sphere );
            EXPECT_TRUE( mapper.getMaps( cubeObj ).cut2origin == fullMapper.getMaps( cubeObj ).cut2origin );
        }
    }
}

// the precut output of either mesh of an operation cutting both, or of both meshes at once, has what cutMesh needs for the boolean's
// exact cut: the cut of the precut mesh by its contours with sortData( the other mesh ) is the boolean's own one, here of a cube
// with a lone contour of a small sphere, as mesh A and as mesh B, also with a rotated rigidB2A, through both overloads (where the Mesh&&
// one copies no mesh: the cut of the other mesh sorts its intersections by the uncut one itself, and with both outputs nothing is cut)
TEST( MRMesh, BooleanPreCutExactCut )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    Mesh sphere = makeSphere( { .radius = 0.2f, .numMeshVertices = 200 } );
    sphere.transform( AffineXf3f::translation( Vector3f( 1.f, -0.4f, 2.f ) ) );
    const auto xf = AffineXf3f::translation( Vector3f( 0.3f, -0.2f, 0.25f ) ) *
        AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1.f, 2.f, 3.f ).normalized(), 0.7f ) );
    const auto op = BooleanOperation::Intersection; // A is filled on the left of its cut, B on the right

    for ( bool swap : { false, true } )
    {
        for ( const AffineXf3f* rigidB2A : { (const AffineXf3f*)nullptr, &xf } )
        {
            // rigidB2A moves the second mesh back in place
            const Mesh& a = swap ? sphere : cube;
            Mesh b = swap ? cube : sphere;
            if ( rigidB2A )
                b.transform( rigidB2A->inverse() );
            BooleanResultMapper mapper;
            const auto full = boolean( a, b, op, { .rigidB2A = rigidB2A, .mapper = &mapper } );
            ASSERT_TRUE( full.valid() ) << full.errorString;

            for ( bool wantA : { true, false } )
            {
                for ( bool wantB : { true, false } )
                {
                    if ( !wantA && !wantB )
                        continue;
                    for ( bool rvalue : { false, true } )
                    {
                        BooleanPreCutResult preA, preB;
                        const BooleanParameters params{ .rigidB2A = rigidB2A, .outPreCutA = wantA ? &preA : nullptr, .outPreCutB = wantB ? &preB : nullptr };
                        const auto res = rvalue ? boolean( Mesh( a ), Mesh( b ), op, params ) : boolean( a, b, op, params );
                        ASSERT_TRUE( res.valid() ) << res.errorString;
                        for ( bool isA : { true, false } )
                        {
                            if ( !( isA ? wantA : wantB ) )
                                continue;
                            SCOPED_TRACE( std::string( swap ? "sphere is A" : "cube is A" ) + ( rigidB2A ? ", rigidB2A" : "" ) +
                                ( wantA && wantB ? ", both outputs" : "" ) + ( isA ? ", output A" : ", output B" ) + ( rvalue ? ", Mesh&&" : "" ) );
                            const auto& pre = isA ? preA : preB;
                            EXPECT_TRUE( pre.mesh == ( isA ? a : b ) );
                            EXPECT_EQ( pre.isMeshA, isA );
                            EXPECT_EQ( pre.meshAVertsNum, a.topology.vertSize() );
                            EXPECT_FALSE( pre.intersections.empty() );
                            EXPECT_EQ( pre.rigidB2A.has_value(), rigidB2A != nullptr );
                            if ( rigidB2A && pre.rigidB2A )
                            {
                                EXPECT_TRUE( *pre.rigidB2A == *rigidB2A );
                            }

                            Mesh cut = pre.mesh;
                            FaceMap new2Old;
                            const auto sortData = pre.sortData( isA ? b : a );
                            const auto cutRes = cutMesh( cut, pre.contours, { .sortData = &sortData, .new2OldMap = &new2Old,
                                .fillPart = isA ? CutMeshParameters::FillPart::Left : CutMeshParameters::FillPart::Right } );
                            EXPECT_TRUE( cutRes.fbsWithContourIntersections.none() );
                            EXPECT_TRUE( new2Old == mapper.getMaps( isA ? BooleanResultMapper::MapObject::A : BooleanResultMapper::MapObject::B ).cut2origin );
                        }
                    }
                }
            }
        }
    }
}

// selfBoolean of a mesh of two parts, a cube and a small sphere piercing a big triangle of it without crossing its edges
// (a lone contour); the parts are only reconnected along their intersection, so the result keeps the sum of their volumes
TEST( MRMesh, SelfBooleanLoneContour )
{
    Mesh mesh = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    Mesh sphere = makeSphere( { .radius = 0.2f, .numMeshVertices = 200 } );
    sphere.transform( AffineXf3f::translation( Vector3f( 1.f, -0.4f, 2.f ) ) );
    const double volume = mesh.volume() + sphere.volume();
    mesh.addMesh( sphere );

    const auto res = selfBoolean( mesh );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_TRUE( res->topology.checkValidity() );
    EXPECT_EQ( res->topology.findNumHoles(), 0 );
    EXPECT_NEAR( res->volume(), volume, 1e-5 * volume );
}

// open fan of three triangles around 0_v: the last one crosses the first one from the shared vertex 0_v to its boundary edge 3_v-4_v;
// the boundary walks from that edge reach 0_v and 2_v, vertices of the first triangle, so the open contour must not be closed through them;
// edge 3_v-4_v crosses the first triangle downward or upward, so the missing face is on the right of the contour's first edge or on the left of its last one
TEST( MRMesh, SelfBooleanContourToBoundary )
{
    for ( float z : { 0.5f, -0.5f } )
    {
        Mesh mesh;
        mesh.points = VertCoords
        {
            { 0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { -0.2f, 0.6f, z },
            { 0.6f, 0.2f, -z }
        };
        Triangulation t =
        {
            { 0_v, 1_v, 2_v },
            { 0_v, 2_v, 3_v },
            { 0_v, 3_v, 4_v }
        };
        mesh.topology = MeshBuilder::fromTriangles( t );

        const auto res = selfBoolean( mesh );
        ASSERT_TRUE( res.has_value() ) << res.error();
        EXPECT_TRUE( res->topology.checkValidity() );
        // contours running out to the boundary are not cut yet
        EXPECT_EQ( res->topology.numValidFaces(), 3 );
        EXPECT_EQ( res->topology.numValidVerts(), 5 );
        for ( auto v : mesh.topology.getValidVerts() )
            EXPECT_LT( ( res->points[v] - mesh.points[v] ).length(), 1e-6f ); // only the round trip through integer coordinates
    }
}

// closed mesh of two parts (apexes 3_v and 6_v) stitched along their common boundary, which passes through all vertices of triangle 0_v, 2_v, 1_v
// of each part, so these two triangles share edges 0_v-2_v and 0_v-1_v, while their edges 1_v-2_v make a multiple edge
static Mesh makeMeshWithMultipleEdge()
{
    Mesh res;
    res.points = VertCoords
    {
        {0.0f,0.0f,0.0f},
        {-0.5f,1.0f,0.0f},
        {0.5f,1.0f,0.0f},
        {0.0f,1.5f,0.5f},
        {-1.0f,1.5f,0.0f},
        {1.0f,1.5f,0.0f}
    };
    Triangulation t =
    {
        { 0_v, 2_v, 1_v },
        { 1_v, 2_v, 3_v },
        { 3_v, 4_v, 1_v },
        { 2_v, 5_v, 3_v },
        { 3_v, 5_v, 4_v }
    };
    res.topology = MeshBuilder::fromTriangles( t );

    Mesh second = res;
    second.points[3_v] = { 0.0f,1.5f,-0.5f };
    auto border = trackRightBoundaryLoop( res.topology, res.topology.findHoleRepresentiveEdges()[0] );
    res.addMeshPart( second, true, { border }, { border } );
    return res;
}

TEST( MRMesh, BooleanMultipleEdgePropogationSort )
{
    const Mesh meshA = makeMeshWithMultipleEdge();

    auto meshB = makeCube( Vector3f::diagonal( 2.0f ) );
    meshB.transform( AffineXf3f::translation( Vector3f( -1.5f, -0.2f, -0.5f ) ) );


    for ( int i = 0; i<int( BooleanOperation::Count ); ++i )
    {
        EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation( i ) ).valid() );
        EXPECT_TRUE( boolean( meshB, meshA, BooleanOperation( i ) ).valid() );
    }
}

// both edges of the multiple edge (see makeMeshWithMultipleEdge) cross a big triangle of the cube at one point inside it, on a lone contour
// that goes through this point twice, from the upper half of the lens to the lower one; the exact predicates of the cut tell apart the two
// crossings defined alike by the third vertices of the triangles of each edge, so every boolean and selfBoolean give valid closed meshes
// with the volumes of the halves of the lens on either side of the cube's side x = 0.1
TEST( MRMesh, BooleanMultipleEdgeThroughLoneFace )
{
    const Mesh lens = makeMeshWithMultipleEdge();
    const Mesh cube = makeCube( Vector3f::diagonal( 20.f ), Vector3f( 0.1f, -9.f, -0.6f ) );
    const double volCube = cube.volume(), volLens = lens.volume();
    EXPECT_NEAR( volCube, 8000., 1e-6 * 8000. );
    EXPECT_NEAR( volLens, 0.25, 1e-6 * 0.25 );
    for ( bool swap : { false, true } )
    {
        const Mesh& a = swap ? lens : cube;
        const Mesh& b = swap ? cube : lens;
        double vols[3] = {};
        // the volumes of the intersection and the union, and of the difference of the cube and the lens or the other way round
        const double expected[3] = { 0.100166672, 8000.149845, swap ? 0.1498333365 : 7999.899845 };
        for ( int i = 0; i < 3; ++i )
        {
            const auto op = std::array{ BooleanOperation::Intersection, BooleanOperation::Union, BooleanOperation::DifferenceAB }[i];
            const auto res = boolean( a, b, op );
            ASSERT_TRUE( res.valid() ) << res.errorString;
            EXPECT_TRUE( res.mesh.topology.checkValidity() );
            EXPECT_EQ( res.mesh.topology.findNumHoles(), 0 );
            vols[i] = res.mesh.volume();
            EXPECT_NEAR( vols[i], expected[i], 1e-6 * expected[i] );
        }
        const double volA = swap ? volLens : volCube, volB = swap ? volCube : volLens;
        EXPECT_NEAR( vols[1] + vols[0], volA + volB, 1e-6 * ( volA + volB ) );
        EXPECT_NEAR( vols[2], volA - vols[0], 1e-6 * ( volA + volB ) );
    }

    Mesh both = cube;
    both.addMesh( lens );
    const auto res = selfBoolean( both );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_TRUE( res->topology.checkValidity() );
    EXPECT_EQ( res->topology.findNumHoles(), 0 );
}

// the tip at 0_v of two triangles with the same vertices (see makeMeshWithMultipleEdge) pokes through a big triangle of the cube:
// edges 0_v-1_v and 0_v-2_v, which both triangles share, cross it at two points, on a closed lone contour of two points; it has zero area
// and is dropped, so the boolean and selfBoolean give valid closed meshes
TEST( MRMesh, BooleanTwoPointLoneContour )
{
    const Mesh lens = makeMeshWithMultipleEdge();
    const Mesh cube = makeCube( Vector3f::diagonal( 20.f ), Vector3f( -3.f, 0.5f, -15.f ) ); // only its side y = 0.5 crosses the lens
    for ( bool swap : { false, true } )
    {
        const Mesh& a = swap ? lens : cube;
        const Mesh& b = swap ? cube : lens;
        for ( auto op : { BooleanOperation::Intersection, BooleanOperation::Union, BooleanOperation::DifferenceAB } )
        {
            const auto res = boolean( a, b, op );
            ASSERT_TRUE( res.valid() ) << res.errorString;
            EXPECT_TRUE( res.mesh.topology.checkValidity() );
            EXPECT_EQ( res.mesh.topology.findNumHoles(), 0 );
        }
    }

    Mesh both = cube;
    both.addMesh( lens );
    const auto res = selfBoolean( both );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_TRUE( res->topology.checkValidity() );
    EXPECT_EQ( res->topology.findNumHoles(), 0 );
}

// a corner of a small cube, or the lowest vertex of a small sphere, lies exactly on a big triangle of a cube, the rest of it outside;
// the edges from that vertex meet the triangle at one point, so simulation of simplicity makes a lone contour of points that are all
// the same in float (here with the big cube as mesh A), or no contact (as mesh B); either way the results are valid closed meshes
// with the volumes of meshes touching at a point
TEST( MRMesh, BooleanVertexOnFace )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    const Vector3f p( 1.f, -0.4f, 2.f ); // on the top z = 2 of the cube, in its triangle below the diagonal y = x
    Mesh corner = makeCube( Vector3f::diagonal( 0.4f ), Vector3f::diagonal( -0.2f ) );
    corner.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::diagonal( -1.f ), Vector3f::minusZ() ) ) ); // a corner down
    Mesh sphere = makeSphere( { .radius = 0.2f, .numMeshVertices = 200 } );
    for ( Mesh* m : { &corner, &sphere } )
    {
        VertId low;
        for ( auto v : m->topology.getValidVerts() )
            if ( !low || m->points[v].z < m->points[low].z )
                low = v;
        m->transform( AffineXf3f::translation( p - m->points[low] ) );
        m->points[low] = p; // exactly
        m->invalidateCaches();
    }

    for ( const Mesh* tip : { &corner, &sphere } )
    {
        for ( bool swap : { false, true } )
        {
            const Mesh& a = swap ? *tip : cube;
            const Mesh& b = swap ? cube : *tip;
            for ( auto op : { BooleanOperation::Intersection, BooleanOperation::Union, BooleanOperation::DifferenceAB } )
            {
                const auto res = boolean( a, b, op );
                ASSERT_TRUE( res.valid() ) << res.errorString;
                EXPECT_TRUE( res.mesh.topology.checkValidity() );
                EXPECT_EQ( res.mesh.topology.findNumHoles(), 0 );
                const double volume = op == BooleanOperation::Union ? a.volume() + b.volume() : op == BooleanOperation::Intersection ? 0 : a.volume();
                EXPECT_NEAR( res.mesh.volume(), volume, 1e-6 * cube.volume() );
            }
        }
    }
}

// a peak of the terrain pokes through a big triangle of the structure without crossing its edges, so the structure is cut along
// a lone contour; far from the origin with the peak just above the structure, all points of the contour are the same in float,
// so only the exact predicates tell how to connect it with the corners of the triangle
TEST( MRMesh, EmbedStructureLoneContour )
{
    for ( float shift : { 0.f, 1000.f } )
    {
        const float peak = shift == 0 ? 2.f : std::nextafter( 1.f, 2.f );
        const auto terrain = makeRegularGridMesh( 21, 21, [] ( size_t, size_t ) { return true; }, [&] ( size_t x, size_t y )
        {
            return Vector3f( shift + float( x ) - 10.f, shift + float( y ) - 10.f, x == 10 && y == 10 ? peak : 0.f );
        } );
        ASSERT_TRUE( terrain.has_value() );
        Mesh structure = makePlane(); // the peak is inside one of its two triangles
        structure.transform( AffineXf3f( Matrix3f::scale( 8.f ), Vector3f( shift + 1.f, shift - 1.f, 1.f ) ) );
        const auto res = embedStructureToTerrain( *terrain, structure, { .fillAngle = PI_F / 4, .cutAngle = PI_F / 4 } );
        ASSERT_TRUE( res.has_value() ) << res.error();
        EXPECT_TRUE( res->topology.checkValidity() );
        EXPECT_EQ( res->topology.findNumHoles(), 1 ); // the border of the terrain
    }
}

// the wall of a small structure above a terrain of two big triangles reaches the terrain inside one of them, so the terrain is cut
// along a lone contour, which the cut connects with the rest of the triangle; every face of the terrain in the result maps to
// the triangle of the input terrain it is in
TEST( MRMesh, EmbedStructureLoneTerrainContour )
{
    Mesh terrain = makePlane();
    terrain.transform( AffineXf3f::linear( Matrix3f::scale( 40.f ) ) );
    Mesh structure = makePlane();
    structure.transform( AffineXf3f::translation( Vector3f( 8.f, -8.f, 0.5f ) ) ); // above the triangle below the diagonal y = x
    const FaceId lower = findProjection( Vector3f( 8.f, -8.f, 0.f ), terrain ).proj.face;
    ASSERT_TRUE( lower.valid() );
    FaceMap new2Old;
    const auto res = embedStructureToTerrain( terrain, structure, { .fillAngle = PI_F / 4, .cutAngle = PI_F / 4, .new2oldFaces = &new2Old } );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_TRUE( res->topology.checkValidity() );
    EXPECT_EQ( res->topology.findNumHoles(), 1 ); // the border of the terrain
    int numTerrainFaces = 0;
    for ( auto f : res->topology.getValidFaces() )
    {
        if ( f >= new2Old.size() )
            continue; // the faces of the structure and its walls come after those of the cut terrain
        ++numTerrainFaces;
        const auto c = res->triCenter( f );
        EXPECT_TRUE( terrain.topology.hasFace( new2Old[f] ) ) << "face " << f;
        EXPECT_EQ( new2Old[f] == lower, c.y < c.x ) << "face " << f;
    }
    EXPECT_GT( numTerrainFaces, 0 );
}

// two triangles sharing two vertices but not the edge between them (multiple edges connect the vertices) can meet only along
// the segment between these vertices, so findSelfCollidingEdgeTrisPrecise skips them as the triangles with a shared edge,
// but still finds all crossings of other edges and triangles
TEST( MRMesh, SelfCollidingEdgeTrisPreciseMultipleEdges )
{
    Mesh mesh = makeMeshWithMultipleEdge();
    ASSERT_EQ( findMultipleEdges( mesh.topology ).value().size(), 1 );

    // all crossings of an edge and a triangle without common vertices, with the edge directed as in findSelfCollidingEdgeTrisPrecise
    auto findAllCrossings = [&mesh]( const ConvertToIntVector & toInt )
    {
        std::vector<std::pair<EdgeId, FaceId>> res;
        auto pvc = [&]( VertId v ) { return PreciseVertCoords{ v, toInt( mesh.points[v] ) }; };
        for ( auto ue : undirectedEdges( mesh.topology ) )
        {
            const EdgeId e = ue;
            const auto o = mesh.topology.org( e ), d = mesh.topology.dest( e );
            for ( auto f : mesh.topology.getValidFaces() )
            {
                if ( mesh.topology.isTriVert( f, o ) || mesh.topology.isTriVert( f, d ) )
                    continue;
                const auto t = mesh.topology.getTriVerts( f );
                if ( auto isect = doTriangleSegmentIntersect( { pvc( t[0] ), pvc( t[1] ), pvc( t[2] ), pvc( o ), pvc( d ) } ) )
                    res.emplace_back( isect.dIsLeftFromABC ? e : e.sym(), f );
            }
        }
        std::sort( res.begin(), res.end() );
        return res;
    };

    // first the closed mesh without self-intersections; then apex 6_v of the second part is moved through the first part,
    // and triangle 1_v, 2_v, 3_v with the multiple edge gets two crossings: its edge 2_v-3_v crosses triangle 5_v, 6_v, 4_v,
    // and edge 4_v-6_v of triangle 4_v, 6_v, 1_v (sharing vertex 1_v with it) crosses it
    for ( bool crossing : { false, true } )
    {
        if ( crossing )
        {
            mesh.points[6_v] = { 1.0f, 1.0f, 0.5f };
            mesh.invalidateCaches();
        }
        const auto toInt = getVectorConverters( mesh ).toInt;
        const auto expected = findAllCrossings( toInt );
        EXPECT_EQ( expected.size(), crossing ? 2 : 0 );

        std::vector<std::pair<EdgeId, FaceId>> found;
        for ( const auto & et : findSelfCollidingEdgeTrisPrecise( mesh, toInt ) )
            found.emplace_back( et.edge, et.tri );
        std::sort( found.begin(), found.end() );
        EXPECT_EQ( found, expected );
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

// the tests of the functions deprecated since cutMesh connects lone contours itself
MR_SUPPRESS_WARNING_PUSH
MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )

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
    const Mesh sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    const auto xf = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), 0.1f ) );
    const auto conv = getVectorConverters( sphere, sphere, &xf );
    const auto orgRes = findCollidingEdgeTrisPrecise( sphere, sphere, conv.toInt, &xf );
    ASSERT_GT( orgRes.size(), 100 );
    auto less = []( const VarEdgeTri & a, const VarEdgeTri & b )
    {
        return std::make_tuple( a.isEdgeATriB(), int( a.edge ), int( a.tri() ) ) < std::make_tuple( b.isEdgeATriB(), int( b.edge ), int( b.tri() ) );
    };
    auto sortedOrgRes = orgRes;
    std::sort( sortedOrgRes.begin(), sortedOrgRes.end(), less );

    // the splits of the triangles of every 200th intersection add less than 1/32 of new vertices, so the update itself is tested;
    // the ones of every 10th intersection add more, and then the function makes new search
    for ( size_t step : { 200, 10 } )
    {
        Mesh meshA = sphere;
        Mesh meshB = sphere;
        auto res = orgRes;

        // split the intersected triangles of both meshes with new vertices inside the meshes, and an intersecting edge of A;
        // several intersections can have the same triangle, then the part of it keeping its id is split again
        const VertId aFirstNewVert( meshA.topology.vertSize() );
        const VertId bFirstNewVert( meshB.topology.vertSize() );
        FaceHashMap aNew2Old, bNew2Old;
        for ( size_t i = 0; i < res.size(); i += step )
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
        const int numVerts = int( meshA.topology.vertSize() + meshB.topology.vertSize() );
        EXPECT_EQ( 32 * ( numVerts - aFirstNewVert - bFirstNewVert ) > numVerts, step == 10 );

        updateCollidingEdgeTrisPrecise( res, meshA, aFirstNewVert, meshB, bFirstNewVert, conv.toInt, &xf );
        auto newRes = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt, &xf );
        std::sort( res.begin(), res.end(), less );
        std::sort( newRes.begin(), newRes.end(), less );
        EXPECT_EQ( res, newRes );
        EXPECT_NE( res, sortedOrgRes );

        // no new vertices
        updateCollidingEdgeTrisPrecise( res, meshA, VertId( meshA.topology.vertSize() ), meshB, VertId( meshB.topology.vertSize() ), conv.toInt, &xf );
        EXPECT_EQ( res, newRes );
    }
}

MR_SUPPRESS_WARNING_POP

} //namespace MR
