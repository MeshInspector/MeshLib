#include <MRMesh/MRContoursCut.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MREdgePaths.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRFillContour.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MR2DContoursTriangulation.h>
#include <cmath>
#include <random>

namespace MR
{

namespace
{

// cutMesh expects every two consecutive intersections of a contour to lie in one face
bool haveCommonFace( const MeshTopology& topology, const OneMeshIntersection& a, const OneMeshIntersection& b )
{
    auto facesOf = [&] ( const OneMeshIntersection& inter )
    {
        std::vector<FaceId> res;
        if ( auto v = std::get_if<VertId>( &inter.primitiveId ) )
        {
            for ( auto e : orgRing( topology, *v ) )
                if ( auto f = topology.left( e ) )
                    res.push_back( f );
        }
        else if ( auto e = std::get_if<EdgeId>( &inter.primitiveId ) )
        {
            for ( auto f : { topology.left( *e ), topology.right( *e ) } )
                if ( f )
                    res.push_back( f );
        }
        else
            res.push_back( std::get<FaceId>( inter.primitiveId ) );
        return res;
    };
    const auto fa = facesOf( a ), fb = facesOf( b );
    return std::any_of( fa.begin(), fa.end(), [&] ( FaceId f ) { return std::find( fb.begin(), fb.end(), f ) != fb.end(); } );
}

// checks offset contours: every closed one ends where it starts, every open one goes from a boundary edge to a boundary edge,
// no step leaves the faces; then cuts the mesh with them; returns the number of open contours
int checkOffsetContoursAndCut( const Mesh& mesh, const OneMeshContours& contours )
{
    const auto& topology = mesh.topology;
    int numOpen = 0;
    for ( size_t c = 0; c < contours.size(); ++c )
    {
        const auto& in = contours[c].intersections;
        EXPECT_GE( in.size(), 2 ) << "contour " << c;
        if ( in.size() < 2 )
            continue;
        for ( size_t i = 0; i + 1 < in.size(); ++i )
        {
            EXPECT_TRUE( haveCommonFace( topology, in[i], in[i + 1] ) ) << "contour " << c << ": intersections " << i << " and " << i + 1 << " have no common face";
        }
        if ( contours[c].closed )
        {
            EXPECT_EQ( in.front().coordinate, in.back().coordinate ) << "contour " << c;
            EXPECT_TRUE( in.front().primitiveId == in.back().primitiveId ) << "contour " << c;
            continue;
        }
        ++numOpen;
        for ( const auto& end : { in.front(), in.back() } )
        {
            auto e = std::get_if<EdgeId>( &end.primitiveId );
            EXPECT_TRUE( e && topology.isBdEdge( *e ) ) << "contour " << c << " is open, but does not end on the boundary";
        }
    }

    // open contours are cut from boundary to boundary
    auto cut = mesh;
    auto res = cutMesh( cut, contours );
    EXPECT_TRUE( cut.topology.checkValidity() );
    EXPECT_TRUE( res.fbsWithContourIntersections.none() );
    EXPECT_EQ( res.resultCut.size(), contours.size() );
    EXPECT_EQ( cut.topology.findNumHoles(), topology.findNumHoles() );
    return numOpen;
}

// 3x3 squares in XY plane, vertex 4 * y + x is at ( x, y ), with faces ( v, v + 1, v + 5 ) and ( v, v + 5, v + 4 ) in square v
Mesh makeSquares3x3()
{
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 4; ++y )
        for ( int x = 0; x < 4; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    for ( int y = 0; y < 3; ++y )
        for ( int x = 0; x < 3; ++x )
        {
            const VertId v( 4 * y + x );
            t.push_back( { v, v + 1, v + 5 } );
            t.push_back( { v, v + 5, v + 4 } );
        }
    mesh.topology = MeshBuilder::fromTriangles( t );
    return mesh;
}

} //anonymous namespace

// the path from the last point back to the first one is empty, and the first path crosses the edge exactly at the last point
TEST( MRMesh, ClosedContourWithEmptyWrapAroundPath )
{
    VertCoords points;
    Triangulation t;
    for ( int y = 0; y < 6; ++y )
        for ( int x = 0; x < 6; ++x )
            points.emplace_back( float( x ), float( y ), 0.0f );
    for ( int y = 0; y < 5; ++y )
        for ( int x = 0; x < 5; ++x )
        {
            const VertId a( y * 6 + x ), b( a + 1 ), c( a + 6 ), d( a + 7 );
            t.push_back( { a, b, d } );
            t.push_back( { a, d, c } );
        }
    Mesh mesh;
    mesh.points = std::move( points );
    mesh.topology = MeshBuilder::fromTriangles( t );

    std::vector<MeshTriPoint> mtps;
    for ( auto p : { Vector3f( 1.5f, 0.9f, 0 ), Vector3f( 1.5f, 2.5f, 0 ), Vector3f( 3.5f, 2.5f, 0 ), Vector3f( 3.5f, 0.9f, 0 ), Vector3f( 1.5f, 1, 0 ) } )
        mtps.push_back( mesh.projectPoint( p ).mtp );
    mtps.push_back( mtps.front() );

    auto contour = surfaceLineToMeshContour( mesh, mtps );
    ASSERT_TRUE( contour.has_value() );
    EXPECT_TRUE( contour->closed );
    for ( const auto& inter : contour->intersections )
        std::visit( [] ( auto id ) { EXPECT_TRUE( id.valid() ); }, inter.primitiveId );

    auto cutCopy = mesh;
    auto res = cutMesh( cutCopy, { *contour } );
    EXPECT_TRUE( res.fbsWithContourIntersections.none() );
    EXPECT_EQ( res.resultCut.size(), 1 );

    // the first point takes over the last intersection of the path to the last point, and must still be the contour start
    mtps.clear();
    for ( auto p : { Vector3f( 1.5f, 0.9f, 0 ), Vector3f( 1.5f, 2.5f, 0 ), Vector3f( 3.5f, 2.5f, 0 ), Vector3f( 3.5f, 1.5f, 0 ), Vector3f( 1.5f, 1, 0 ) } )
        mtps.push_back( mesh.projectPoint( p ).mtp );
    mtps.push_back( mtps.front() );
    std::vector<int> pivots;
    contour = surfaceLineToMeshContour( mesh, mtps, SearchPathSettings{}, &pivots );
    ASSERT_TRUE( contour.has_value() );
    ASSERT_EQ( pivots.size(), mtps.size() );
    EXPECT_EQ( pivots.front(), 0 );
    EXPECT_EQ( pivots.back(), 0 );
    int lastPivot = -1;
    for ( size_t i = 0; i + 1 < pivots.size(); ++i )
    {
        if ( pivots[i] < 0 )
            continue;
        EXPECT_GT( pivots[i], lastPivot );
        lastPivot = pivots[i];
    }
}

TEST( MRMesh, BooleanIntersectionsSort )
{
    Mesh meshA;
    meshA.points = VertCoords
    {
        { 8.95297337f, 14.3548975f,-0.212119192f },
        { 8.98828983f, 14.3914976f,-0.198161319f },
        { 8.92162418f, 14.4169340f,-0.203402281f },
        { 8.95297337f, 14.4501600f,-0.191835344f }
    };
    Triangulation tA =
    {
        { 0_v, 1_v, 3_v },
        { 0_v, 3_v, 2_v }
    };
    meshA.topology = MeshBuilder::fromTriangles( tA );

    Mesh meshB;
    meshB.points = VertCoords
    {
        { 8.91892719f, 14.3419390f, -0.208497435f },
        { 8.99423218f, 14.4023476f, -0.208966389f },
        { 9.00031281f, 14.4126110f, -0.209267750f },
        { 8.99934673f, 14.4161797f, -0.209171638f },
        { 8.91623878f, 14.3510427f, -0.205425277f }
    };
    Triangulation tB =
    {
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 4_v },
        { 2_v, 3_v, 4_v }
    };
    meshB.topology = MeshBuilder::fromTriangles( tB );
    auto converters = getVectorConverters( meshA, meshB );
    auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt );
    auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    OneMeshContours meshAContours, meshBContours;
    getOneMeshIntersectionContours( meshA, meshB, contours, &meshAContours, &meshBContours, converters );

    SortIntersectionsData dataForA{meshB,contours,converters.toInt,nullptr,meshA.topology.vertSize(),false};

    Vector3f aNorm;
    for ( auto f : meshA.topology.getValidFaces() )
        aNorm += meshA.dirDblArea( f );
    aNorm = aNorm.normalized();
    CutMeshParameters params;
    params.sortData = &dataForA;
    cutMesh( meshA, meshAContours, params );

    for ( auto f : meshA.topology.getValidFaces() )
        EXPECT_TRUE( dot( meshA.dirDblArea( f ), aNorm ) > 0.0f );
}

// a closed contour going along a boundary edge must not reference the missing face behind it
TEST( MRMesh, ClosedContourAlongBoundaryEdge )
{
    Mesh mesh;
    mesh.points = VertCoords{ { 0, 0, 0 }, { 1, 0, 0.3f }, { 0, 1, 0 }, { 1, 1, 0.3f } };
    mesh.topology = MeshBuilder::fromTriangles( Triangulation{ { 0_v, 1_v, 2_v }, { 1_v, 3_v, 2_v } } );

    std::vector<MeshTriPoint> mtps;
    for ( auto p : { Vector3f( 0, 0.8f, 0 ), Vector3f( 0, 0, 0.005f ), Vector3f( 0, 0.7f, 0 ), Vector3f( 1.3f, 0.2f, 0.36f ) } )
        mtps.push_back( mesh.projectPoint( p ).mtp );
    mtps.push_back( mtps.front() );

    auto contour = surfaceLineToMeshContour( mesh, mtps );
    ASSERT_TRUE( contour.has_value() );
    EXPECT_GE( contour->intersections.size(), 4u );
    for ( const auto& inter : contour->intersections )
        std::visit( [] ( auto id ) { EXPECT_TRUE( id.valid() ); }, inter.primitiveId );

    auto res = cutMesh( mesh, { *contour } );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_TRUE( res.fbsWithContourIntersections.none() );
    EXPECT_EQ( res.resultCut.size(), 1 );
}

// the contour goes along boundary edge W-V and turns back in V to a point on the same edge
TEST( MRMesh, CutMeshContourTurnsBackOnBoundary )
{
    // 4x2 squares in XY plane, vertex 5 * y + x is at ( x, y )
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 3; ++y )
        for ( int x = 0; x < 5; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    for ( int y = 0; y < 2; ++y )
    {
        for ( int x = 0; x < 4; ++x )
        {
            const VertId v( 5 * y + x );
            t.push_back( { v, v + 1, v + 6 } );
            t.push_back( { v, v + 6, v + 5 } );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );

    const VertId w( 1 ), v( 2 );
    const auto wv = mesh.topology.findEdge( w, v );
    ASSERT_TRUE( mesh.topology.isBdEdge( wv ) );
    const MeshTriPoint mtpV( MeshEdgePoint( mesh.topology.findEdge( v, 8_v ), 0.0f ) );
    // otherwise V and the next point are taken as two points on the same edge, and V is dropped
    ASSERT_NE( mtpV.onEdge( mesh.topology ).e.undirected(), wv.undirected() );
    const MeshTriPoint mtpW = mesh.toTriPoint( w );
    const MeshTriPoint nearV( MeshEdgePoint( wv, 0.9f ) );
    const MeshTriPoint inside = findProjection( Vector3f( 1.3f, 1.6f, 0.0f ), mesh ).mtp;

    auto contour = surfaceLineToMeshContour( mesh, { mtpW, mtpV, nearV, inside, mtpW } );
    ASSERT_TRUE( contour.has_value() );
    ASSERT_TRUE( contour->closed );
    for ( const auto& inter : contour->intersections )
    {
        // V can be in a face here, but it must be a face of W-V, shared with the neighbor intersections
        if ( auto f = std::get_if<FaceId>( &inter.primitiveId ); f && inter.coordinate == mesh.points[v] )
        {
            ASSERT_TRUE( *f == mesh.topology.left( wv ) || *f == mesh.topology.right( wv ) );
        }
    }

    auto res = cutMesh( mesh, { *contour }, { .forceFill = true } );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_TRUE( res.fbsWithContourIntersections.none() );
    EXPECT_EQ( res.resultCut.size(), 1 );
}

// the contour goes along the boundary through corner vertex C, whose boundary edges have the hole between them:
// the vertex before C and the point after it on the other boundary edge of C have no common face
TEST( MRMesh, ContourThroughBoundaryCorner )
{
    // 4x2 squares in XY plane, vertex 5 * y + x is at ( x, y ); corner C = 0 has faces (0,1,6) and (0,6,5)
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 3; ++y )
        for ( int x = 0; x < 5; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    for ( int y = 0; y < 2; ++y )
    {
        for ( int x = 0; x < 4; ++x )
        {
            const VertId v( 5 * y + x );
            t.push_back( { v, v + 1, v + 6 } );
            t.push_back( { v, v + 6, v + 5 } );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );
    const auto& topology = mesh.topology;

    const VertId a( 1 ), c( 0 ), b( 5 );
    const auto cb = topology.findEdge( c, b );
    ASSERT_TRUE( topology.isBdEdge( cb ) );
    ASSERT_TRUE( topology.isBdEdge( topology.findEdge( c, a ) ) );
    const MeshTriPoint mtpA = mesh.toTriPoint( a );
    const MeshTriPoint onCB( MeshEdgePoint( cb, 0.5f ) );
    const MeshTriPoint before = findProjection( Vector3f( 1.6f, 0.3f, 0.0f ), mesh ).mtp;
    const MeshTriPoint after = findProjection( Vector3f( 0.3f, 1.5f, 0.0f ), mesh ).mtp;
    const MeshTriPoint cOnOtherEdge = mesh.toTriPoint( c );
    ASSERT_NE( cOnOtherEdge.onEdge( topology ).e.undirected(), cb.undirected() );
    // C stored so that onEdge() gives C-B: the cleanup of duplicates drops C as a second point on the edge of the next point,
    // and the path from A to the point on C-B goes inside the mesh
    MeshTriPoint cOnCB;
    for ( auto e : orgRing( topology, c ) )
        if ( MeshTriPoint( e, TriPointf( 0, 0 ) ).onEdge( topology ).e.undirected() == cb.undirected() )
            cOnCB = MeshTriPoint( e, TriPointf( 0, 0 ) );
    ASSERT_EQ( cOnCB.inVertex( topology ), c );
    for ( const auto& mtpC : { cOnOtherEdge, cOnCB } )
    {
        // forward: vertex A, C, the point on C-B (the vertex-edge test dropped C); backward: the same points in reverse order
        for ( bool forward : { true, false } )
        {
            SCOPED_TRACE( std::string( mtpC.e == cOnCB.e ? "C on C-B" : "C on another edge" ) + ( forward ? ", forward" : ", backward" ) );
            auto contour = forward ?
                surfaceLineToMeshContour( mesh, { before, mtpA, mtpC, onCB, after, before } ) :
                surfaceLineToMeshContour( mesh, { after, onCB, mtpC, mtpA, before, after } );
            ASSERT_TRUE( contour.has_value() );
            ASSERT_TRUE( contour->closed );
            // cutMesh relies on it
            for ( size_t i = 0; i + 1 < contour->intersections.size(); ++i )
            {
                EXPECT_TRUE( haveCommonFace( topology, contour->intersections[i], contour->intersections[i + 1] ) )
                    << "intersections " << i << " and " << i + 1 << " have no common face";
            }

            auto cut = mesh;
            auto res = cutMesh( cut, { *contour }, { .forceFill = true } );
            EXPECT_TRUE( cut.topology.checkValidity() );
            EXPECT_TRUE( res.fbsWithContourIntersections.none() );
            EXPECT_EQ( res.resultCut.size(), 1 );
            // an edge across the hole at C would split the boundary
            EXPECT_EQ( cut.topology.findNumHoles(), 1 );
        }
    }
}

// the contour touches a boundary edge of ear tip E (boundary vertex with one face) and returns into that face
TEST( MRMesh, ContourTouchesBoundaryAtEarTip )
{
    // 3x3 squares in XY plane, vertex 4 * y + x is at ( x, y ); the first square has diagonal 1-4, so E = 0 has the only face (0,1,4)
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 4; ++y )
        for ( int x = 0; x < 4; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    t.push_back( { 0_v, 1_v, 4_v } );
    t.push_back( { 1_v, 5_v, 4_v } );
    for ( int y = 0; y < 3; ++y )
    {
        for ( int x = y == 0 ? 1 : 0; x < 3; ++x )
        {
            const VertId v( 4 * y + x );
            t.push_back( { v, v + 1, v + 5 } );
            t.push_back( { v, v + 5, v + 4 } );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );
    const auto e01 = mesh.topology.findEdge( 0_v, 1_v ), e04 = mesh.topology.findEdge( 0_v, 4_v );
    ASSERT_TRUE( mesh.topology.isBdEdge( e01 ) && mesh.topology.isBdEdge( e04 ) );
    const MeshTriPoint inEar = findProjection( Vector3f( 0.3f, 0.3f, 0.0f ), mesh ).mtp;
    const MeshTriPoint inside = findProjection( Vector3f( 1.5f, 1.6f, 0.0f ), mesh ).mtp;

    // three closed contours, each touches a boundary edge of E and returns into the ear face:
    // 1. V4 -> point on E-1 -> inside the ear face;
    // 2. inside the ear face -> point on E-4 -> V1;
    // 3. inside the ear face -> point on E-4 -> point on E-1.
    // On master the point on the boundary edge was kept with the hole as its face towards the intersection before it (1)
    // or after it (2, 3), and cutMesh joined them through the hole: the cut split the boundary loop (1, 2) or filled the outer hole (3)
    for ( const auto& mtps : { std::vector<MeshTriPoint>{ mesh.toTriPoint( 4_v ), MeshTriPoint( MeshEdgePoint( e01, 0.5f ) ), inEar, inside },
                               std::vector<MeshTriPoint>{ inEar, MeshTriPoint( MeshEdgePoint( e04, 0.5f ) ), mesh.toTriPoint( 1_v ), inside },
                               std::vector<MeshTriPoint>{ inEar, MeshTriPoint( MeshEdgePoint( e04, 0.5f ) ), MeshTriPoint( MeshEdgePoint( e01, 0.5f ) ), inside } } )
    {
        auto closedMtps = mtps;
        closedMtps.push_back( mtps.front() );
        auto contour = surfaceLineToMeshContour( mesh, closedMtps );
        ASSERT_TRUE( contour.has_value() );
        ASSERT_TRUE( contour->closed );

        Mesh cut = mesh;
        auto res = cutMesh( cut, { *contour }, { .forceFill = true } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.none() );
        EXPECT_EQ( res.resultCut.size(), 1 );
        EXPECT_EQ( cut.topology.findNumHoles(), 1 );
        EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f );
    }
}

// the contour goes along edge A-B of face (A,B,C), then back into that face to a point on edge A-C
TEST( MRMesh, CutMeshContourTurnsInsideFace )
{
    const Mesh mesh = makeSquares3x3();
    const auto& topology = mesh.topology;

    // inside the mesh, and with B-C on the boundary
    for ( auto [a, b, c, q] : { std::tuple{ 10_v, 5_v, 6_v, Vector3f( 2.3f, 1.8f, 0.0f ) }, std::tuple{ 6_v, 1_v, 2_v, Vector3f( 2.3f, 0.6f, 0.0f ) } } )
    {
        // the step A-B removes face (A,B,C) on its left, the step B -> point on A-C goes in it again
        const auto ab = topology.findEdge( a, b );
        ASSERT_EQ( topology.dest( topology.next( ab ) ), c );
        const auto ac = topology.findEdge( a, c );
        ASSERT_EQ( topology.dest( topology.prev( ac ) ), b );
        OneMeshContour contour;
        contour.closed = true;
        contour.intersections = {
            { a, mesh.points[a] },
            { b, mesh.points[b] },
            { ac, 0.5f * ( mesh.points[a] + mesh.points[c] ) },
            { topology.left( ac ), q },
            { a, mesh.points[a] } };

        Mesh cut = mesh;
        auto res = cutMesh( cut, { contour }, { .forceFill = true } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.none() );
        EXPECT_EQ( res.resultCut.size(), 1 );
        // the part of (A,B,C) between B, C and the new point was not filled
        EXPECT_EQ( cut.topology.findNumHoles(), 1 );
        EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f );
    }
}

// a closed contour leaves a mesh vertex into a face and comes back from the same face,
// so both its edges at the vertex go in the place of that face, in the order of their directions
TEST( MRMesh, CutMeshContourLeavesAndEntersVertexInOneFace )
{
    const Mesh mesh = makeSquares3x3();

    // face (5, 6, 10) and two points in it
    const auto v = 5_v;
    const auto f = mesh.topology.left( mesh.topology.findEdge( v, 6_v ) );
    const Vector3f p( 1.8f, 1.2f, 0.0f ), q( 1.8f, 1.6f, 0.0f );
    for ( bool ccw : { true, false } )
    {
        OneMeshContour contour;
        contour.closed = true;
        contour.intersections = {
            { v, mesh.points[v] },
            { f, ccw ? p : q },
            { f, ccw ? q : p },
            { v, mesh.points[v] } };

        Mesh cut = mesh;
        auto res = cutMesh( cut, { contour } );
        EXPECT_TRUE( cut.topology.checkValidity() ) << ccw;
        EXPECT_TRUE( res.fbsWithContourIntersections.none() ) << ccw;
        ASSERT_EQ( res.resultCut.size(), 1 );
        EXPECT_EQ( cut.topology.findNumHoles(), 1 ) << ccw;
        EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f ) << ccw;
        // the triangle (v, p, q) is on the left of a counterclockwise contour
        const auto inside = fillContourLeft( cut.topology, res.resultCut[0] );
        EXPECT_NEAR( cut.area( ccw ? inside : cut.topology.getValidFaces() - inside ), 0.16f, 1e-5f ) << ccw;
    }
}

// a closed contour goes from a mesh vertex to a point in a face and back to the vertex, then around a loop in the next face or not,
// starting at every point (at the touched point too: x, v, ..., v, x and x, v, x): the cut goes along one edge to the point and back,
// with no hole of two edges between them to fill
TEST( MRMesh, CutMeshContourTouchesPointFromVertex )
{
    const Mesh mesh = makeSquares3x3();

    // faces (5, 6, 10) and (5, 10, 9)
    const auto v = 5_v;
    const auto f = mesh.topology.left( mesh.topology.findEdge( v, 6_v ) );
    const auto g = mesh.topology.left( mesh.topology.findEdge( v, 10_v ) );
    const OneMeshIntersection iv{ v, mesh.points[v] }, ix{ f, Vector3f( 1.8f, 1.2f, 0.0f ) };
    for ( const auto& points : { std::vector{ iv, ix, iv, OneMeshIntersection{ g, Vector3f( 1.6f, 1.8f, 0.0f ) },
        OneMeshIntersection{ g, Vector3f( 1.2f, 1.6f, 0.0f ) } }, std::vector{ iv, ix } } )
    {
        const int n = int( points.size() );
        for ( int start = 0; start < n; ++start )
        {
            SCOPED_TRACE( testing::Message() << n << " points, start " << start );
            OneMeshContour contour;
            contour.closed = true;
            for ( int i = 0; i <= n; ++i )
                contour.intersections.push_back( points[( start + i ) % n] );

            Mesh cut = mesh;
            auto res = cutMesh( cut, { contour } );
            EXPECT_TRUE( cut.topology.checkValidity() );
            EXPECT_TRUE( res.fbsWithContourIntersections.none() );
            ASSERT_EQ( res.resultCut.size(), 1 );
            ASSERT_EQ( res.resultCut[0].size(), n );
            // the path edges from v to x and back
            const int vx = ( n - start ) % n;
            EXPECT_EQ( res.resultCut[0][( vx + 1 ) % n], res.resultCut[0][vx].sym() );
            EXPECT_EQ( cut.topology.findNumHoles(), 1 );
            EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f );
            // only the points of the contour other than v are new vertices
            EXPECT_EQ( cut.topology.numValidVerts(), mesh.topology.numValidVerts() + ( n == 2 ? 1 : 3 ) );
        }
    }
}

// open contours touching a point from a mesh vertex, going along an edge of the mesh and back, or ending inside faces,
// where the edges to the end are a spike in the hole of their face; that hole is filled with the rest of the face, whichever part is filled
TEST( MRMesh, CutMeshOpenContourTouchesPoint )
{
    const Mesh mesh = makeSquares3x3();
    const auto& topology = mesh.topology;

    // faces (5, 6, 10) and (5, 10, 9), with edge 5 -> 10 between them, f on its right
    const auto v = 5_v;
    const auto f = topology.left( topology.findEdge( v, 6_v ) );
    const auto e = topology.findEdge( v, 10_v );
    const auto g = topology.left( e );
    ASSERT_EQ( topology.right( e ), f );
    const OneMeshIntersection iv{ v, mesh.points[v] }, ix{ f, Vector3f( 1.8f, 1.2f, 0.0f ) }, ix2{ f, Vector3f( 1.8f, 1.6f, 0.0f ) },
        iy{ g, Vector3f( 1.2f, 1.6f, 0.0f ) }, ip{ 1_v, mesh.points[1_v] }, iq{ 9_v, mesh.points[9_v] }, ie{ e, Vector3f( 1.5f, 1.5f, 0.0f ) };
    const std::pair<Vector<OneMeshIntersection>, int> cases[] = {
        { { ip, iv, ix, iv, iq }, 1 },
        { { iv, ix, iv }, 1 },
        { { ip, iv, ix }, 1 },
        { { ip, iv, ix, ix2 }, 2 },
        { { ip, iv, ip }, 0 },
        { { iv, ix }, 1 },
        { { ix, ie, iy }, 3 },
        // a point touched from both ends: two new vertices at one position, two spikes in one hole
        { { ix, iv, ix }, 2 } };
    for ( const auto& [points, numNewVerts] : cases )
    {
        for ( auto part : { CutMeshParameters::FillPart::Both, CutMeshParameters::FillPart::Left, CutMeshParameters::FillPart::Right } )
        {
            SCOPED_TRACE( testing::Message() << points.size() << " points, fill part " << int( part ) );
            OneMeshContour contour;
            contour.intersections = points;

            Mesh cut = mesh;
            auto res = cutMesh( cut, { contour }, { .fillPart = part } );
            EXPECT_TRUE( cut.topology.checkValidity() );
            EXPECT_TRUE( res.fbsWithContourIntersections.none() );
            ASSERT_EQ( res.resultCut.size(), 1 );
            EXPECT_EQ( res.resultCut[0].size() + 1, points.size() );
            EXPECT_EQ( cut.topology.numValidVerts(), mesh.topology.numValidVerts() + numNewVerts );
            if ( part != CutMeshParameters::FillPart::Both )
                continue;
            EXPECT_EQ( cut.topology.findNumHoles(), 1 );
            EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f );
        }
    }
}

// a contour inside one triangle (a lone contour) is connected with the rest of the triangle by the cut, near the origin and far from it,
// also where the contour fills most of the triangle; a closed contour is an island in the hole of the triangle, an open one a slit
TEST( MRMesh, CutMeshLoneContour )
{
    // uniform values in [-1, 1] straight from the generator, whose output the standard specifies (unlike that of the distributions),
    // so they are the same on every platform; the points made of them with std::cos and std::sin can differ in their last bits there,
    // which the tolerances absorb
    std::mt19937 gen( 1 );
    auto random = [&] { return 2.0 * gen() / std::mt19937::max() - 1; };
    auto randomVec = [&]
    {
        Vector3d res;
        for ( int i = 0; i < 3; ++i )
            res[i] = random();
        return res;
    };
    for ( double dist : { 0.0, 1e5 } )
    {
        for ( float fill : { 0.5f, 0.9f } )
        {
            for ( int trial = 0; trial < 50; ++trial )
            {
                // an equilateral triangle in a random plane, so that every axis is dropped by the projection in some trials
                const auto base = dist * randomVec();
                const auto u = randomVec().normalized();
                const auto v = cross( u, randomVec() ).normalized();
                VertCoords points;
                for ( int i = 0; i < 3; ++i )
                    points.push_back( Vector3f( base + 10.0 * ( std::cos( 2 * PI * i / 3 ) * u + std::sin( 2 * PI * i / 3 ) * v ) ) );
                const auto mesh = Mesh::fromTriangles( std::move( points ), { { 0_v, 1_v, 2_v } } );

                // the triangle shrunk around its centroid to the given part of its area, with four points on every side
                const auto e = mesh.topology.edgeWithLeft( 0_f );
                const float k = std::sqrt( fill ), m = ( 1 - k ) / 3;
                const std::array<Vector2f, 3> corners{ Vector2f( m, m ), Vector2f( m + k, m ), Vector2f( m, m + k ) };
                std::vector<MeshTriPoint> mtps;
                for ( int i = 0; i < 3; ++i )
                    for ( int j = 0; j < 4; ++j )
                    {
                        const auto b = corners[i] + ( corners[( i + 1 ) % 3] - corners[i] ) * ( j / 4.0f );
                        mtps.push_back( MeshTriPoint( e, TriPointf( b.x, b.y ) ) );
                    }
                auto closed = mtps;
                closed.push_back( mtps.front() );
                // the closed contour, the same without its last segment, and a slit of one segment (a hole loop of 2 edges)
                for ( const auto& line : { closed, mtps, std::vector<MeshTriPoint>{ mtps[0], mtps[6] } } )
                {
                    auto contour = surfaceLineToMeshContour( mesh, line );
                    ASSERT_TRUE( contour.has_value() );

                    Mesh cut = mesh;
                    NewEdgesMap new2OldEdges;
                    auto res = cutMesh( cut, { *contour }, { .new2oldEdgesMap = &new2OldEdges } );
                    EXPECT_TRUE( cut.topology.checkValidity() );
                    EXPECT_TRUE( res.fbsWithContourIntersections.none() )
                        << "distance " << dist << ", fill " << fill << ", trial " << trial << ", points " << line.size();
                    EXPECT_EQ( cut.topology.findNumHoles(), 1 ); // the island or the slit is connected and filled on both sides
                    EXPECT_NEAR( cut.area(), mesh.area(), 1e-3 * mesh.area() );
                    // every new edge, as a chord connecting the contour with the sides, maps to the triangle (surface offset relies on it)
                    for ( auto ue = UndirectedEdgeId( mesh.topology.undirectedEdgeSize() ); ue < cut.topology.undirectedEdgeSize(); ++ue )
                    {
                        const auto it = new2OldEdges.map.find( ue );
                        ASSERT_NE( it, new2OldEdges.map.end() );
                        EXPECT_FALSE( new2OldEdges.splitEdges.test( ue ) );
                        EXPECT_EQ( it->second, 0 ); // 0_f
                    }
                    // the cut goes along the contour
                    ASSERT_EQ( res.resultCut.size(), 1 );
                    const auto& path = res.resultCut[0];
                    ASSERT_EQ( path.size() + 1, contour->intersections.size() );
                    EXPECT_TRUE( isEdgePath( cut.topology, path ) );
                    for ( int i = 0; i < path.size(); ++i )
                        EXPECT_EQ( cut.orgPnt( path[i] ), contour->intersections[i].coordinate );
                }
            }
        }
    }
}

// a lone contour crossing itself (a bowtie inside one triangle of a closed mesh) cannot be connected with the triangle;
// the cut reports the triangle as bad and keeps it, so that its sides make no hole whichever faces are filled
TEST( MRMesh, CutMeshSelfCrossingLoneContour )
{
    const auto mesh = makeCube();
    const auto f = 0_f;
    const auto e = mesh.topology.edgeWithLeft( f );
    // side 0-1 crosses side 2-3
    std::vector<MeshTriPoint> mtps;
    for ( const auto& b : { Vector2f( 0.2f, 0.2f ), Vector2f( 0.5f, 0.4f ), Vector2f( 0.5f, 0.2f ), Vector2f( 0.2f, 0.4f ), Vector2f( 0.2f, 0.2f ) } )
        mtps.push_back( MeshTriPoint( e, TriPointf( b.x, b.y ) ) );
    const auto contour = surfaceLineToMeshContour( mesh, mtps );
    ASSERT_TRUE( contour.has_value() );

    for ( bool forceFill : { false, true } )
    {
        Mesh cut = mesh;
        const auto res = cutMesh( cut, { *contour }, { .forceFill = forceFill } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.test( f ) );
        EXPECT_TRUE( cut.topology.hasFace( f ) );
        EXPECT_EQ( cut.topology.left( e ), f );
        // the two sides of the island are left without faces, or filled
        EXPECT_EQ( cut.topology.findNumHoles(), forceFill ? 0 : 2 );
    }
}

// a lone contour inside a triangle of zero area (its corners on one line) has no plane to connect it with the triangle in;
// without the exact sorting data, the cut reports the triangle as bad and keeps it; a contour through the triangle along that line
// with four points in it, which the plain cut takes, makes it bad under force fill, with no plane to find their crossings in either
TEST( MRMesh, CutMeshLoneContourInZeroAreaFace )
{
    // corner 2_v of 1_f is on its side from 1_v to 0_v
    const auto mesh = Mesh::fromTriangles( { Vector3f( 0, 0, 0 ), Vector3f( 10, 10, 0 ), Vector3f( 4, 4, 0 ), Vector3f( 0, 10, 0 ) },
        { { 0_v, 1_v, 3_v }, { 1_v, 0_v, 2_v } } );
    const auto e = mesh.topology.edgeWithLeft( 1_f );
    OneMeshContour island;
    for ( float t : { 1.f, 3.f, 7.f, 1.f } )
        island.intersections.push_back( { .primitiveId = 1_f, .coordinate = Vector3f( t, t, 0 ) } );
    island.closed = true;
    // from the side of 1_f between 0_v and 2_v through two points inside it to its side between 0_v and 1_v, then across 0_f to the border
    const OneMeshContour through{ .intersections = { { .primitiveId = mesh.topology.findEdge( 0_v, 2_v ), .coordinate = Vector3f( 2, 2, 0 ) },
        { .primitiveId = 1_f, .coordinate = Vector3f( 3, 3, 0 ) }, { .primitiveId = 1_f, .coordinate = Vector3f( 5, 5, 0 ) },
        { .primitiveId = mesh.topology.findEdge( 0_v, 1_v ), .coordinate = Vector3f( 6, 6, 0 ) },
        { .primitiveId = mesh.topology.findEdge( 3_v, 1_v ), .coordinate = Vector3f( 5, 10, 0 ) } } };

    for ( bool forceFill : { false, true } )
    {
        Mesh cut = mesh;
        const auto res = cutMesh( cut, { island }, { .forceFill = forceFill } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.test( 1_f ) );
        EXPECT_TRUE( cut.topology.hasFace( 1_f ) );
        EXPECT_EQ( cut.topology.left( e ), 1_f );
        // the border of the mesh, and the two sides of the island unless they are filled
        EXPECT_EQ( cut.topology.findNumHoles(), forceFill ? 1 : 3 );

        cut = mesh;
        const auto bad = cutMesh( cut, { through }, { .forceFill = forceFill } ).fbsWithContourIntersections;
        EXPECT_EQ( bad.count(), forceFill ? 1 : 0 );
        EXPECT_EQ( bad.test( 1_f ), forceFill );
    }
}

// a small square pierces a big triangle of a cube without crossing its edges, so the cube is cut along an open lone contour (a slit)
// of three points on one line, where the sides and the diagonal of the square cross the triangle; the steps of the boolean before its cut
// (as in the terrain embedding) give the exact sorting data, with the square as mesh A, and as mesh B, whose vertex ids the predicates shift
TEST( MRMesh, CutMeshLoneSlitExact )
{
    const auto cube = makeCube( Vector3f::diagonal( 4.f ), Vector3f::diagonal( -2.f ) );
    // the square of side 0.4 in the plane y = -0.4, centered on the top z = 2 of the cube, in its triangle below the diagonal y = x
    auto square = makePlane();
    square.transform( AffineXf3f( Matrix3f( { 0.4f, 0, 0 }, { 0, 0, -0.4f }, { 0, 0.4f, 0 } ), Vector3f( 1.f, -0.4f, 2.f ) ) );

    for ( bool squareIsA : { true, false } )
    {
        const auto& a = squareIsA ? square : cube;
        const auto& b = squareIsA ? cube : square;
        const auto conv = getVectorConverters( a, b );
        const auto contours = orderIntersectionContours( a.topology, b.topology, findCollidingEdgeTrisPrecise( a, b, conv.toInt ) );
        OneMeshContours cubeContours;
        getOneMeshIntersectionContours( a, b, contours, squareIsA ? nullptr : &cubeContours, squareIsA ? &cubeContours : nullptr, conv );
        ASSERT_EQ( cubeContours.size(), 1 );
        ASSERT_EQ( cubeContours[0].intersections.size(), 3 );
        EXPECT_FALSE( cubeContours[0].closed );
        const SortIntersectionsData sortData{ square, contours, conv.toInt, nullptr, a.topology.vertSize(), squareIsA };

        Mesh cut = cube;
        const auto res = cutMesh( cut, cubeContours, { .sortData = &sortData } );
        EXPECT_TRUE( res.fbsWithContourIntersections.none() ) << "square is " << ( squareIsA ? "A" : "B" );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_EQ( cut.topology.findNumHoles(), 0 ); // the slit is connected and filled on both sides
        EXPECT_NEAR( cut.area(), cube.area(), 1e-6 * cube.area() );
        ASSERT_EQ( res.resultCut.size(), 1 );
        EXPECT_EQ( res.resultCut[0].size(), 2 );
        EXPECT_TRUE( isEdgePath( cut.topology, res.resultCut[0] ) );
    }
}

namespace
{

// the square [0, 10]x[0, 10] in the plane z = 0 of two triangles: 0_f below its diagonal from 0_v to 2_v, 1_f above it
Mesh makeSquareOfTwoTriangles()
{
    return Mesh::fromTriangles( { Vector3f( 0, 0, 0 ), Vector3f( 10, 0, 0 ), Vector3f( 10, 10, 0 ), Vector3f( 0, 10, 0 ) },
        { { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v } } );
}

// an island inside triangle 0_f of the square: the closed contour around the triangle (6, 2), (8, 2), (8, 4)
OneMeshContour makeIslandInSquare( const Mesh& square )
{
    std::vector<MeshTriPoint> mtps;
    for ( const auto& p : { Vector3f( 6, 2, 0 ), Vector3f( 8, 2, 0 ), Vector3f( 8, 4, 0 ), Vector3f( 6, 2, 0 ) } )
        mtps.push_back( square.toTriPoint( 0_f, p ) );
    auto res = surfaceLineToMeshContour( square, mtps );
    EXPECT_TRUE( res.has_value() && res->closed && res->intersections.size() == 4 );
    return res.value_or( OneMeshContour{} );
}

// the point of a contour on side ab of the square, which the contour crosses from the right of ab to its left
OneMeshIntersection onSquareSide( const Mesh& square, VertId a, VertId b, const Vector3f& p )
{
    return { .primitiveId = square.topology.findEdge( a, b ), .coordinate = p };
}

// cuts the square by the contours, which cross none of one another, and checks that the cut is valid without bad faces and keeps
// the area, every island is connected, and the maps take every new face, and every new edge inside a triangle, to the triangle it is in
void checkSquareCut( const Mesh& square, const OneMeshContours& contours )
{
    Mesh cut = square;
    FaceMap new2Old;
    NewEdgesMap new2OldEdges;
    const auto res = cutMesh( cut, contours, { .new2OldMap = &new2Old, .new2oldEdgesMap = &new2OldEdges } );
    const auto& tp = cut.topology;
    EXPECT_TRUE( tp.checkValidity() );
    EXPECT_TRUE( res.fbsWithContourIntersections.none() );
    EXPECT_EQ( tp.findNumHoles(), 1 ); // the border of the square
    EXPECT_EQ( MeshComponents::getNumComponents( cut ), 1 ); // no island is left unconnected
    EXPECT_NEAR( cut.area(), square.area(), 1e-5 * square.area() );
    auto triangleOf = [] ( const Vector3f& p ) { return p.y < p.x ? 0_f : 1_f; };
    for ( auto f : tp.getValidFaces() )
    {
        EXPECT_EQ( f < new2Old.size() ? new2Old[f] : FaceId(), triangleOf( cut.triCenter( f ) ) ) << "face " << f;
    }
    for ( const auto& [ue, old] : new2OldEdges.map )
    {
        const EdgeId e = ue;
        const auto mid = 0.5f * ( cut.orgPnt( e ) + cut.destPnt( e ) );
        if ( new2OldEdges.splitEdges.test( ue ) || std::abs( mid.x - mid.y ) < 1e-5f )
            continue; // a part of an old edge, or on the diagonal
        EXPECT_EQ( FaceId( old ), triangleOf( mid ) ) << "edge " << e;
    }
}

} //anonymous namespace

// an island inside triangle 0_f of the square, and an open contour from the side of the square in triangle 1_f to the middle of
// the diagonal: the cut of the diagonal fills 0_f again with the end of the open contour as a new corner before the island is
// connected, so the island has nothing to connect with there, and 0_f is reported as bad instead of the island being left unconnected
TEST( MRMesh, CutMeshLoneContourInRefilledFace )
{
    const auto square = makeSquareOfTwoTriangles();
    const auto island = makeIslandInSquare( square );
    const OneMeshContour open{ .intersections = {
        onSquareSide( square, 3_v, 0_v, Vector3f( 0, 5, 0 ) ), onSquareSide( square, 2_v, 0_v, Vector3f( 5, 5, 0 ) ) } };
    for ( bool islandFirst : { false, true } )
    {
        Mesh cut = square;
        const auto res = cutMesh( cut, islandFirst ? OneMeshContours{ island, open } : OneMeshContours{ open, island } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.test( 0_f ) ) << "island first " << islandFirst;
    }
}

// a contour along the diagonal from 0_v to 2_v (0_f on its right) or back (0_f on its left), with an island inside 0_f, which the cut
// connects with the rest of 0_f after removing it, or with a contour cutting off the corner of 0_f at 1_v, in either order: the faces
// filled in 0_f map to it, also on the right of the diagonal, where they took the face on its left (1_f) before
TEST( MRMesh, CutMeshContourAlongSideMaps )
{
    const auto square = makeSquareOfTwoTriangles();
    auto along = [&] ( VertId a, VertId b )
    {
        return OneMeshContour{ .intersections = {
            { .primitiveId = a, .coordinate = square.points[a] }, { .primitiveId = b, .coordinate = square.points[b] } } };
    };
    const OneMeshContour corner{ .intersections = {
        onSquareSide( square, 0_v, 1_v, Vector3f( 8, 0, 0 ) ), onSquareSide( square, 2_v, 1_v, Vector3f( 10, 2, 0 ) ) } };
    const std::pair<const char*, OneMeshContour> alongs[] = { { "along 0-2", along( 0_v, 2_v ) }, { "along 2-0", along( 2_v, 0_v ) } };
    const std::pair<const char*, OneMeshContour> others[] = { { "island", makeIslandInSquare( square ) }, { "corner", corner } };
    for ( const auto& [alongName, alongContour] : alongs )
    {
        for ( const auto& [otherName, other] : others )
        {
            for ( bool alongFirst : { true, false } )
            {
                SCOPED_TRACE( std::string( alongName ) + ( alongFirst ? " before " : " after " ) + otherName );
                checkSquareCut( square, alongFirst ? OneMeshContours{ alongContour, other } : OneMeshContours{ other, alongContour } );
            }
        }
    }
}

// an island inside triangle 0_f of the square, and a contour from the side of the square in triangle 1_f through 0_f to its side
// on the border of the square, in either order: the cut connects the island with the part of 0_f around it on the float path
TEST( MRMesh, CutMeshLoneContourWithCrossingContour )
{
    const auto square = makeSquareOfTwoTriangles();
    const auto island = makeIslandInSquare( square );
    const OneMeshContour crossing{ .intersections = { onSquareSide( square, 3_v, 0_v, Vector3f( 0, 2, 0 ) ),
        onSquareSide( square, 2_v, 0_v, Vector3f( 1.5f, 1.5f, 0 ) ), onSquareSide( square, 1_v, 0_v, Vector3f( 6, 0, 0 ) ) } };
    for ( bool islandFirst : { true, false } )
    {
        SCOPED_TRACE( islandFirst ? "island first" : "island second" );
        checkSquareCut( square, islandFirst ? OneMeshContours{ island, crossing } : OneMeshContours{ crossing, island } );
    }
}

TEST( MRMesh, MeshCollidePrecise )
{
    const auto meshA = makeTorus( 1.1f, 0.5f, 8, 8 );
    auto meshB = makeTorus( 1.1f, 0.5f, 8, 8 );
    meshB.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), Vector3f { 0.1f, 0.8f, 0.2f } ) ) );

    auto conv = getVectorConverters( meshA, meshB );

    const auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt );
    EXPECT_EQ( intersections.size(), 152 );

    const auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    EXPECT_EQ( contours.size(), 4 );
    EXPECT_TRUE( contours[0].size() == 69 || // Apple Debug
                 contours[0].size() == 71 );
    EXPECT_TRUE( contours[1].size() == 9 ||  // Apple Debug
                 contours[1].size() == 7 );
    EXPECT_TRUE( contours[2].size() == 69 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 contours[2].size() == 71 ); // with FMA instruction (modern compilers for ARM)
    EXPECT_TRUE( contours[3].size() == 9 ||  // without FMA instruction (default settings for x86 or old compilers for ARM)
                 contours[3].size() == 7 );  // with FMA instruction (modern compilers for ARM)

    for ( const auto & c : contours )
    {
        // check that contour is closed
        EXPECT_EQ( c.front(), c.back() );

        // check intersections' order
        for ( int i = 0; i + 1 < c.size(); ++i )
        {
            const auto & curr = c[i];
            const auto& currEdgeTopology = curr.isEdgeATriB() ? meshA.topology : meshB.topology;
            const auto& currTriTopology = curr.isEdgeATriB() ? meshB.topology : meshA.topology;
            const auto & next = c[i+1];
            if ( curr.isEdgeATriB() == next.isEdgeATriB() )
            {
                EXPECT_TRUE( currEdgeTopology.next( curr.edge ) == next.edge
                    || currEdgeTopology.prev( curr.edge.sym() ).sym() == next.edge );
                EXPECT_EQ( curr.tri(), next.tri() );
            }
            else
            {
                EXPECT_EQ( currEdgeTopology.left( curr.edge ), next.tri() );
                bool found = false;
                for ( EdgeId e : leftRing( currTriTopology, curr.tri() ) )
                    found = found || e.sym() == next.edge;
                EXPECT_TRUE( found );
            }
        }

        // check edges' orientation
        for ( const auto & vet : c )
        {
            if ( vet.isEdgeATriB() )
            {
                const auto pl = meshB.getPlane3d( vet.tri() );
                EXPECT_LE( pl.distance( Vector3d{ meshA.orgPnt( vet.edge ) } ), 0 );
                EXPECT_GE( pl.distance( Vector3d{ meshA.destPnt( vet.edge ) } ), 0 );
            }
            else
            {
                const auto pl = meshA.getPlane3d( vet.tri() );
                EXPECT_GE( pl.distance( Vector3d{ meshB.orgPnt( vet.edge ) } ), 0 );
                EXPECT_LE( pl.distance( Vector3d{ meshB.destPnt( vet.edge ) } ), 0 );
            }
        }
    }

    OneMeshContours meshAContours, meshBContours;
    getOneMeshIntersectionContours( meshA, meshB, contours, &meshAContours, &meshBContours, conv );
    EXPECT_EQ( meshAContours.size(), 4 );
    EXPECT_EQ( meshBContours.size(), 4 );

    size_t posCount = 0;
    for ( const auto& contour : meshAContours )
        posCount += contour.intersections.size();
    EXPECT_EQ( posCount, 156 );

    // same for self-intersections
    auto mergedMesh = meshA;
    mergedMesh.addMesh( meshB );
    conv = getVectorConverters( mergedMesh );
    const auto selfIntersections = findSelfCollidingEdgeTrisPrecise( mergedMesh, conv.toInt );
    EXPECT_EQ( selfIntersections.size(), 152 );

    const auto selfContours = orderSelfIntersectionContours( mergedMesh.topology, selfIntersections );
    EXPECT_EQ( selfContours.size(), 8 );
    for ( int i = 0; i < 8; i += 2 )
    {
        const auto & even = selfContours[i];
        const auto & odd = selfContours[i+1];
        EXPECT_EQ( even.size(), odd.size() );
        EXPECT_GT( even.size(), 0 );
        EXPECT_TRUE( even[0].isEdgeATriB() );
        EXPECT_FALSE( odd[0].isEdgeATriB() );
        for ( int j = 0; j < even.size(); ++j )
        {
            const auto& ei = even[j];
            const auto& oi = odd[even.size() - j - 1];
            EXPECT_EQ( ei.tri(), oi.tri() );
            EXPECT_EQ( ei.isEdgeATriB(), !oi.isEdgeATriB() );
            EXPECT_EQ( ei.edge, oi.edge.sym() );
        }
    }
    EXPECT_TRUE( selfContours[0].size() == 69 || // Apple Debug
                 selfContours[0].size() == 71 );
    EXPECT_TRUE( selfContours[2].size() == 69 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 selfContours[2].size() == 71 ); // with FMA instruction (modern compilers for ARM)
    EXPECT_TRUE( selfContours[4].size() == 9 ||  // without FMA instruction (default settings for x86 or old compilers for ARM)
                 selfContours[4].size() == 7 );  // with FMA instruction (modern compilers for ARM)
    EXPECT_TRUE( selfContours[6].size() == 9 ||  // Apple Debug
                 selfContours[6].size() == 7 );

    for ( const auto & c : selfContours )
    {
        // check that contour is closed
        EXPECT_EQ( c.front(), c.back() );

        // check intersections' order
        for ( int i = 0; i + 1 < c.size(); ++i )
        {
            const auto & curr = c[i];
            const auto & next = c[i+1];
            if ( curr.isEdgeATriB() == next.isEdgeATriB() )
            {
                EXPECT_TRUE( mergedMesh.topology.next( curr.edge ) == next.edge
                    || mergedMesh.topology.prev( curr.edge.sym() ).sym() == next.edge );
                EXPECT_EQ( curr.tri(), next.tri() );
            }
            else
            {
                EXPECT_EQ( mergedMesh.topology.left( curr.edge ), next.tri() );
                bool found = false;
                for ( EdgeId e : leftRing( mergedMesh.topology, curr.tri() ) )
                    found = found || e.sym() == next.edge;
                EXPECT_TRUE( found );
            }
        }

        // check edges' orientation
        for ( const auto & vet : c )
        {
            const auto pl = mergedMesh.getPlane3d( vet.tri() );
            if ( vet.isEdgeATriB() )
            {
                EXPECT_LE( pl.distance( Vector3d{ mergedMesh.orgPnt( vet.edge ) } ), 0 );
                EXPECT_GE( pl.distance( Vector3d{ mergedMesh.destPnt( vet.edge ) } ), 0 );
            }
            else
            {
                EXPECT_GE( pl.distance( Vector3d{ mergedMesh.orgPnt( vet.edge ) } ), 0 );
                EXPECT_LE( pl.distance( Vector3d{ mergedMesh.destPnt( vet.edge ) } ), 0 );
            }
        }
    }
}

// new vertices of the cut get ids from vertSize(), so invalid trailing vertices must not change the offset
TEST( MRMesh, SurfaceOffsetWithInvalidTrailingVerts )
{
    VertCoords points;
    Triangulation t;
    for ( int y = 0; y < 6; ++y )
        for ( int x = 0; x < 6; ++x )
            points.emplace_back( float( x ), float( y ), 0.0f );
    for ( int y = 0; y < 5; ++y )
        for ( int x = 0; x < 5; ++x )
        {
            const VertId a( y * 6 + x ), b( a + 1 ), c( a + 6 ), d( a + 7 );
            t.push_back( { a, b, d } );
            t.push_back( { a, d, c } );
        }
    Mesh mesh;
    mesh.points = std::move( points );
    mesh.topology = MeshBuilder::fromTriangles( t );

    std::vector<MeshTriPoint> mtps;
    for ( auto p : { Vector3f( 1.3f, 1.4f, 0 ), Vector3f( 3.6f, 1.2f, 0 ), Vector3f( 3.7f, 3.4f, 0 ), Vector3f( 1.2f, 3.6f, 0 ) } )
        mtps.push_back( mesh.projectPoint( p ).mtp );
    mtps.push_back( mtps.front() );

    auto expected = offsetSurfaceLineToMeshContours( mesh, mtps, 0.3f );
    ASSERT_TRUE( expected.has_value() );
    ASSERT_FALSE( expected->empty() );

    mesh.topology.vertResize( mesh.topology.vertSize() + 3 );
    mesh.points.resize( mesh.topology.vertSize() );
    auto res = offsetSurfaceLineToMeshContours( mesh, mtps, 0.3f );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->size(), expected->size() );
    for ( size_t i = 0; i < res->size(); ++i )
    {
        ASSERT_EQ( ( *res )[i].intersections.size(), ( *expected )[i].intersections.size() );
        for ( size_t j = 0; j < ( *res )[i].intersections.size(); ++j )
            EXPECT_EQ( ( *res )[i].intersections[j].coordinate, ( *expected )[i].intersections[j].coordinate );
    }
}

// the offset of a contour near the open mesh boundary: one isoline is cut by the boundary
TEST( MRMesh, SurfaceOffsetContoursReachBoundary )
{
    // 10x10 squares in XY plane, vertex 11 * y + x is at ( x, y )
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 11; ++y )
        for ( int x = 0; x < 11; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    for ( int y = 0; y < 10; ++y )
    {
        for ( int x = 0; x < 10; ++x )
        {
            const VertId v( 11 * y + x );
            t.push_back( { v, v + 1, v + 12 } );
            t.push_back( { v, v + 12, v + 11 } );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );

    // square along grid lines with corners ( 1, 1 ) and ( 7, 7 )
    std::vector<MeshTriPoint> mtps;
    for ( int v : { 12, 18, 84, 78, 12 } )
        mtps.push_back( mesh.toTriPoint( VertId( v ) ) );

    // the inner isoline is closed, the outer one goes from boundary y = 0 to boundary x = 0
    auto res = offsetSurfaceLineToMeshContours( mesh, mtps, 2.3f );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->size(), 2 );
    EXPECT_EQ( checkOffsetContoursAndCut( mesh, *res ), 1 );
}

// like the reported case of #7167: a closed contour around a flat patch with a wavy open border, partly beyond the border,
// so that many of its points are projected on the border; the outer offset is cut by the border into open pieces
TEST( MRMesh, SurfaceOffsetAlongOpenBorder )
{
    // 20x20 squares on [-10,10]^2 with alternating diagonals; the faces inside a wavy disk are kept
    const int n = 21;
    VertCoords points;
    for ( int y = 0; y < n; ++y )
        for ( int x = 0; x < n; ++x )
            points.emplace_back( float( x - 10 ), float( y - 10 ), 0.0f );
    Triangulation t;
    for ( int y = 0; y + 1 < n; ++y )
    {
        for ( int x = 0; x + 1 < n; ++x )
        {
            const VertId a( y * n + x ), b( a + 1 ), c( a + n ), d( a + n + 1 );
            const bool flip = ( x + y ) % 2 != 0;
            for ( ThreeVertIds tri : { flip ? ThreeVertIds{ a, b, d } : ThreeVertIds{ a, b, c }, flip ? ThreeVertIds{ a, d, c } : ThreeVertIds{ b, d, c } } )
            {
                const auto cen = ( points[tri[0]] + points[tri[1]] + points[tri[2]] ) / 3.0f;
                const float rim = 8.0f * ( 1 + 0.08f * std::cos( 6 * std::atan2( cen.y, cen.x ) ) );
                if ( cen.length() < rim )
                    t.push_back( tri );
            }
        }
    }
    Mesh mesh;
    mesh.points = std::move( points );
    mesh.topology = MeshBuilder::fromTriangles( t );
    ASSERT_EQ( mesh.topology.findNumHoles(), 1 );

    std::vector<MeshTriPoint> mtps;
    int numOnBorder = 0;
    for ( int i = 0; i < 90; ++i )
    {
        const float angle = 2 * PI_F * float( i ) / 90;
        mtps.push_back( mesh.projectPoint( 8.2f * Vector3f( std::cos( angle ), std::sin( angle ), 0.0f ) ).mtp );
        if ( mtps.back().isBd( mesh.topology ) )
            ++numOnBorder;
    }
    ASSERT_GT( numOnBorder, 10 );
    mtps.push_back( mtps.front() );

    auto res = offsetSurfaceLineToMeshContours( mesh, mtps, 1.0f );
    ASSERT_TRUE( res.has_value() );
    const int numOpen = checkOffsetContoursAndCut( mesh, *res );
    EXPECT_GT( numOpen, 0 );
    EXPECT_LT( numOpen, int( res->size() ) ); // the inner offset is closed
}

// an open contour across triangle 0_f of the square at y = 2, and a contour from the bottom side up to a point inside 0_f and back,
// or to a zigzag of 33 points there, so that 0_f has 35 segments with the first contour, which the search tests by the sweep on the float
// coordinates: with the points above the first contour, they cross twice, which leaves valid hole loops, so the plain cut does not
// report it, while under force fill the face of the crossings is bad; with the points below it, the contours do not meet, and the segments
// of the second one only touch
TEST( MRMesh, ForceFillCrossingsEven )
{
    const auto square = makeSquareOfTwoTriangles();
    const OneMeshContour across{ .intersections = {
        onSquareSide( square, 1_v, 2_v, Vector3f( 10, 2, 0 ) ), onSquareSide( square, 0_v, 2_v, Vector3f( 2, 2, 0 ) ) } };
    // from the bottom side up to n points from x = 4.5 to 5.5 (one at x = 5), every other one a quarter below the top, and back
    auto tent = [&] ( float top, int n )
    {
        OneMeshContour res{ .intersections = { onSquareSide( square, 0_v, 1_v, Vector3f( 4, 0, 0 ) ) } };
        for ( int i = 0; i < n; ++i )
            res.intersections.push_back( { .primitiveId = 0_f, .coordinate = Vector3f( 5 + ( i - n / 2 ) / 32.f, top - ( i % 2 ) / 4.f, 0 ) } );
        res.intersections.push_back( onSquareSide( square, 1_v, 0_v, Vector3f( 6, 0, 0 ) ) );
        return res;
    };
    for ( int n : { 1, 33 } )
    {
        for ( float top : { 1.5f, 3.0f } )
        {
            Mesh cut = square;
            EXPECT_TRUE( cutMesh( cut, { across, tent( top, n ) } ).fbsWithContourIntersections.none() );
            cut = square;
            const auto forced = cutMesh( cut, { across, tent( top, n ) }, { .forceFill = true } ).fbsWithContourIntersections;
            EXPECT_EQ( forced.count(), top < 2 ? 0 : 1 );
            EXPECT_EQ( forced.test( 0_f ), top > 2 );
        }
    }
}

// under force fill, a contour crossing itself inside triangle 0_f makes it bad: in one piece of three segments, across the end
// of a closed contour, or in two visits of the face; visiting the face again without a crossing, or touching at a mesh vertex, is fine
TEST( MRMesh, ForceFillCrossingsOneFace )
{
    const auto square = makeSquareOfTwoTriangles();
    auto in = [] ( FaceId f, float x, float y ) { return OneMeshIntersection{ .primitiveId = f, .coordinate = Vector3f( x, y, 0 ) }; };
    auto side = [&] ( VertId a, VertId b, float x, float y ) { return onSquareSide( square, a, b, Vector3f( x, y, 0 ) ); };
    auto check = [&] ( const OneMeshContours& contours, bool bad )
    {
        Mesh cut = square;
        const auto forced = cutMesh( cut, contours, { .forceFill = true } ).fbsWithContourIntersections;
        EXPECT_EQ( forced.count(), bad ? 1 : 0 );
        EXPECT_EQ( forced.test( 0_f ), bad );
    };
    check( { { .intersections = { side( 0_v, 1_v, 4, 0 ), in( 0_f, 8, 3 ), in( 0_f, 8, 1 ), in( 0_f, 5, 3 ), side( 0_v, 2_v, 4, 4 ) } } }, true );
    check( { { .intersections = { in( 0_f, 6, 2 ), side( 0_v, 2_v, 5, 5 ), in( 1_f, 2, 6 ), side( 2_v, 0_v, 3, 3 ), in( 0_f, 8, 4 ), in( 0_f, 6, 2 ) },
        .closed = true } }, true );
    for ( bool cross : { true, false } )
        check( { { .intersections = { side( 0_v, 1_v, 4, 0 ), side( 0_v, 2_v, 5, 5 ), in( 1_f, 2, 6 ), side( 2_v, 0_v, 3, 3 ),
            cross ? side( 2_v, 1_v, 10, 5 ) : side( 1_v, 0_v, 2, 0 ) } } }, cross );
    // two contours through mesh vertex 2_v only touch there, also with a third one from the bottom side up to a point inside 0_f and back,
    // which crosses the second one twice with the point left of it, or meets none with the point right of it: then 0_f, removed at 2_v
    // before the search, has more than three points
    const OneMeshIntersection corner{ .primitiveId = 2_v, .coordinate = Vector3f( 10, 10, 0 ) };
    const OneMeshContour viaCorner[] = { { .intersections = { side( 0_v, 1_v, 4, 0 ), corner, side( 0_v, 3_v, 0, 6 ) } },
        { .intersections = { side( 0_v, 1_v, 7, 0 ), corner, side( 0_v, 3_v, 0, 3 ) } } };
    check( { viaCorner[0], viaCorner[1] }, false );
    for ( bool cross : { true, false } )
        check( { viaCorner[0], viaCorner[1],
            { .intersections = { side( 0_v, 1_v, 8, 0 ), in( 0_f, cross ? 6.5f : 9.5f, 2 ), side( 1_v, 0_v, 9, 0 ) } } }, cross );
}

namespace
{

// the bad faces of meshes A and B by their forced boolean
std::array<FaceBitSet, 2> forcedBooleanBadFaces( const Mesh& meshA, const Mesh& meshB, const AffineXf3f* rigidB2A = nullptr )
{
    auto res = boolean( meshA, meshB, BooleanOperation::Union, { .rigidB2A = rigidB2A, .forceCut = true } );
    return { std::move( res.meshABadContourFaces ), std::move( res.meshBBadContourFaces ) };
}

// two overlapping boxes from z0 to z1 above the middle of the unit cube, whose contours on the top of the cube cross twice
Mesh makeTwoBoxes( float z0, float z1 )
{
    auto res = makeCube( Vector3f( 0.4f, 0.4f, z1 - z0 ), Vector3f( 0.2f, 0.2f, z0 ) );
    res.addMesh( makeCube( Vector3f( 0.4f, 0.4f, z1 - z0 ), Vector3f( 0.4f, 0.4f, z0 ) ) );
    return res;
}

// two cylinders of radius 0.1 with the given number of sides from z = 0.5 to 1.5 around c0 and c1, every one crossing the top of the unit
// cube by its 2 * sides vertical and diagonal side edges
Mesh makeTwoCylinders( int sides, const Vector2f& c0, const Vector2f& c1 )
{
    Mesh res;
    for ( const auto& c : { c0, c1 } )
    {
        auto cylinder = makeCylinder( 0.1f, 1, sides );
        cylinder.transform( AffineXf3f::translation( Vector3f( c.x, c.y, 0.5f ) ) );
        res.addMesh( cylinder );
    }
    return res;
}

} //anonymous namespace

// the top of a cube crossed by two overlapping boxes, whose two closed contours there cross twice, on the exact predicates: axis-aligned,
// where the coordinates leave most pairs open, rotated, with the cube cut as mesh B in other coordinates, and with the boxes standing
// on the top, in its plane
TEST( MRMesh, ForceFillCrossingsBoolean )
{
    const auto cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    auto expectTop = [&] ( const FaceBitSet& bad, const Mesh& mesh, const AffineXf3f& xf )
    {
        EXPECT_EQ( bad.count(), 2 );
        for ( auto f : bad )
            EXPECT_NEAR( xf( mesh.triCenter( f ) ).z, 1, 1e-5f );
    };
    const auto boxes = makeTwoBoxes( 0.5f, 1.5f );
    for ( bool rotated : { false, true } )
    {
        SCOPED_TRACE( rotated ? "rotated" : "axis-aligned" );
        auto b = boxes;
        if ( rotated )
            b.transform( AffineXf3f::xfAround( Matrix3f::rotation( Vector3f( 0.1f, 0.2f, 1 ).normalized(), 0.3f ), Vector3f::diagonal( 0.5f ) ) );
        expectTop( forcedBooleanBadFaces( cube, b )[0], cube, {} );
    }
    // a single box crosses nothing
    EXPECT_TRUE( forcedBooleanBadFaces( cube, makeCube( Vector3f( 0.4f, 0.4f, 1 ), Vector3f( 0.2f, 0.2f, 0.5f ) ) )[0].none() );

    const AffineXf3f xf( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.7f ), Vector3f( 0.3f, -0.2f, 0.1f ) );
    auto local = cube;
    local.transform( xf.inverse() );
    expectTop( forcedBooleanBadFaces( boxes, local, &xf )[1], local, xf );

    expectTop( forcedBooleanBadFaces( cube, makeTwoBoxes( 1, 1.5f ) )[0], cube, {} );
}

// the top of a cube crossed by two cylinders through its diagonal, so that their contours are not lone ones (the plain cut reports the face
// of crossing lone contours already): overlapping cylinders, whose contours cross twice inside the triangle below the diagonal, where they
// have 22 segments with 8 sides, which are tested in pairs, or 154 with 64 sides, which the sweep tests, both on the exact predicates, make
// that triangle bad under force fill, while the plain cut does not report it, as the contours leave valid hole loops there; separate
// cylinders make nothing bad
TEST( MRMesh, ForceFillCrossingsCylinders )
{
    const auto cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    for ( int sides : { 8, 64 } )
    {
        for ( bool overlap : { true, false } )
        {
            SCOPED_TRACE( testing::Message() << sides << " sides, " << ( overlap ? "overlapping" : "separate" ) );
            // the first cylinder is mostly below the diagonal, the second one is on it, by the first one or further along the diagonal
            const auto cylinders = makeTwoCylinders( sides, { 0.55f, 0.48f }, overlap ? Vector2f( 0.52f, 0.51f ) : Vector2f( 0.77f, 0.76f ) );
            EXPECT_TRUE( boolean( cube, cylinders, BooleanOperation::Union ).meshABadContourFaces.none() );
            const auto bad = forcedBooleanBadFaces( cube, cylinders );
            EXPECT_EQ( bad[0].count(), overlap ? 1 : 0 );
            for ( auto f : bad[0] )
            {
                const auto c = cube.triCenter( f );
                EXPECT_NEAR( c.z, 1, 1e-5f );
                EXPECT_GT( c.x, c.y );
            }
            EXPECT_TRUE( bad[1].none() );
        }
    }
}

// the float overload of hasCrossing on more than 32 segments, which the sweep tests: 30 vertical segments, a chain of two segments over
// the tops of the first two, the third one again and reversed, and a segment from a point to itself, all of which only touch, with
// a segment from the top of the first vertical one to the bottom of the second one, touching them only, or of the sixth one, crossing four
TEST( MRMesh, HasCrossingFloatSweep )
{
    VertCoords points;
    Vector<TwoVertIds> segments;
    for ( int i = 0; i < 30; ++i )
    {
        segments.push_back( { VertId( 2 * i ), VertId( 2 * i + 1 ) } );
        points.emplace_back( float( i ), 0.f, 0.f );
        points.emplace_back( float( i ), 10.f, 0.f );
    }
    points.emplace_back( 0.5f, 12.f, 0.f );
    segments.insert( segments.end(), { { 1_v, 60_v }, { 60_v, 3_v }, { 4_v, 5_v }, { 5_v, 4_v }, { 6_v, 6_v } } );
    const auto cache = PlanarTriangulation::makeSweepLineCache();
    for ( bool cross : { true, false } )
    {
        auto s = segments;
        s.push_back( { 1_v, cross ? 10_v : 2_v } );
        EXPECT_EQ( PlanarTriangulation::hasCrossing( s, points, Vector3f::plusZ(), cache.get() ), cross );
    }
}

// a flat rectangle of two triangles and the same triangles reversed, with the same vertices (a closed double-sided sheet, whose diagonals
// make a multiple edge), across the top of the unit cube inside one triangle of it, which the plain cut reports bad: the contours of its sides
// there coincide, and the current crossing search reports that triangle too, since the predicates cannot tell apart the crossings
// of the diagonals of both sides, defined alike, and their failure makes the face bad, although the segments only touch there
TEST( MRMesh, ForceFillDoubleSidedSheet )
{
    const auto cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    const auto sheet = Mesh::fromTriangles( { Vector3f( 0.3f, 0.15f, 0.5f ), Vector3f( 0.7f, 0.15f, 0.5f ), Vector3f( 0.7f, 0.15f, 1.5f ),
        Vector3f( 0.3f, 0.15f, 1.5f ) }, { { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v }, { 0_v, 2_v, 1_v }, { 0_v, 3_v, 2_v } } );
    const auto plain = boolean( cube, sheet, BooleanOperation::Union );
    EXPECT_EQ( plain.meshABadContourFaces.count(), 1 );
    const auto forced = boolean( cube, sheet, BooleanOperation::Union, { .forceCut = true } );
    ASSERT_TRUE( forced.valid() ) << forced.errorString;
    EXPECT_TRUE( forced.mesh.topology.checkValidity() );
    EXPECT_EQ( forced.meshABadContourFaces.count(), 1 );
    EXPECT_EQ( forced.meshABadContourFaces.find_first(), plain.meshABadContourFaces.find_first() );
    EXPECT_TRUE( forced.meshBBadContourFaces.none() );
}

} //namespace MR
