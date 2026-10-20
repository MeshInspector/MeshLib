#include <MRMesh/MRContoursCut.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRPlane3.h>

namespace MR
{

// the path from the last point back to the first one is empty, and the first path crosses the edge exactly at the last point
TEST( MRMesh, ClosedContourWithEmptyWrapAroundPath )
{
    std::vector<Vector3f> points;
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
    meshA.points = std::vector<Vector3f>
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
    meshB.points = std::vector<Vector3f>
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
    mesh.points = std::vector<Vector3f>{ { 0, 0, 0 }, { 1, 0, 0.3f }, { 0, 1, 0 }, { 1, 1, 0.3f } };
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

    auto res = cutMesh( mesh, { *contour }, { .forceFillMode = CutMeshParameters::ForceFill::All } );
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
                const auto fs = facesOf( contour->intersections[i] ), fn = facesOf( contour->intersections[i + 1] );
                EXPECT_TRUE( std::any_of( fs.begin(), fs.end(), [&] ( FaceId f ) { return std::find( fn.begin(), fn.end(), f ) != fn.end(); } ) )
                    << "intersections " << i << " and " << i + 1 << " have no common face";
            }

            auto cut = mesh;
            auto res = cutMesh( cut, { *contour }, { .forceFillMode = CutMeshParameters::ForceFill::All } );
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
        auto res = cutMesh( cut, { *contour }, { .forceFillMode = CutMeshParameters::ForceFill::All } );
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
    // 3x3 squares in XY plane, vertex 4 * y + x is at ( x, y )
    Mesh mesh;
    Triangulation t;
    for ( int y = 0; y < 4; ++y )
        for ( int x = 0; x < 4; ++x )
            mesh.points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
    for ( int y = 0; y < 3; ++y )
    {
        for ( int x = 0; x < 3; ++x )
        {
            const VertId v( 4 * y + x );
            t.push_back( { v, v + 1, v + 5 } );
            t.push_back( { v, v + 5, v + 4 } );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );
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
        auto res = cutMesh( cut, { contour }, { .forceFillMode = CutMeshParameters::ForceFill::All } );
        EXPECT_TRUE( cut.topology.checkValidity() );
        EXPECT_TRUE( res.fbsWithContourIntersections.none() );
        EXPECT_EQ( res.resultCut.size(), 1 );
        // the part of (A,B,C) between B, C and the new point was not filled
        EXPECT_EQ( cut.topology.findNumHoles(), 1 );
        EXPECT_NEAR( cut.area(), mesh.area(), 1e-5f );
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
    std::vector<Vector3f> points;
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

} //namespace MR
