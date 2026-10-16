#include <MRMesh/MRContoursCut.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MROneMeshContours.h>
#include <MRMesh/MRMeshTriPoint.h>
#include <MRMesh/MRMeshProject.h>

namespace MR
{

namespace
{

void expectValidIds( const OneMeshContour& c )
{
    for ( const auto& inter : c.intersections )
        std::visit( [] ( auto id ) { EXPECT_TRUE( id.valid() ); }, inter.primitiveId );
}

// projects the points on the mesh, closes the contour and offsets it along the surface
void checkSurfaceOffset( const Mesh& mesh, const std::vector<Vector3f>& points, float offset )
{
    std::vector<MeshTriPoint> mtps;
    for ( const auto& p : points )
        mtps.push_back( mesh.projectPoint( p ).mtp );
    mtps.push_back( mtps.front() );

    auto contour = convertMeshTriPointsToMeshContour( mesh, mtps );
    ASSERT_TRUE( contour.has_value() );
    expectValidIds( *contour );

    auto res = convertMeshTriPointsSurfaceOffsetToMeshContours( mesh, mtps, offset );
    ASSERT_TRUE( res.has_value() );
    for ( const auto& c : *res )
        expectValidIds( c );
}

} //anonymous namespace

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

// closed contour whose wrap-around path is empty and whose first point is next to the mesh border (#7167)
TEST( MRMesh, SurfaceOffsetContourStartsNearBorder )
{
    Mesh mesh;
    mesh.points = std::vector<Vector3f>
    {
        { 9.03348255f, -0.000231372789f, 0.114755362f },
        { 9.3584547f, 0.00135293894f, 0.0209646747f },
        { 9.67840195f, 0.00390690891f, -0.0749887154f },
        { 9.99948978f, -0.000927287678f, -0.163206339f },
        { 9.02869606f, 0.325817913f, 0.108836338f },
        { 9.35372066f, 0.323594779f, 0.0198833253f },
        { 9.67575359f, 0.321152329f, -0.0711208284f },
        { 10.0010557f, 0.318600386f, -0.154788226f },
        { 9.3554306f, 0.647063792f, 0.016750833f },
        { 9.99775791f, 0.648480415f, -0.130402312f },
    };
    Triangulation t =
    {
        { 0_v, 1_v, 4_v },
        { 1_v, 5_v, 4_v },
        { 1_v, 2_v, 6_v },
        { 1_v, 6_v, 5_v },
        { 2_v, 3_v, 6_v },
        { 3_v, 7_v, 6_v },
        { 4_v, 5_v, 8_v },
        { 5_v, 6_v, 8_v },
        { 6_v, 7_v, 9_v },
    };
    mesh.topology = MeshBuilder::fromTriangles( t );
    checkSurfaceOffset( mesh, {
        { 9.51442242f, 0.0025979476f, -0.0258106645f },
        { 9.51996422f, 0.0139834071f, -0.0273628272f },
        { 9.52384377f, 0.00267314631f, -0.028635893f },
        { 9.52620029f, 0.00269196415f, -0.0293428861f },
    }, 1.70261554f );
}

// contour running along the mesh border: the cut leaves holes with spike edges (#7167)
TEST( MRMesh, SurfaceOffsetContourAlongBorder )
{
    Mesh mesh;
    mesh.points = std::vector<Vector3f>
    {
        { 3.09503722f, -0.014356399f, 0.0114405388f },
        { 3.43319631f, 0.0118827643f, -0.0905694813f },
        { 3.77109528f, 0.0146662584f, -0.181916535f },
        { 4.14991045f, -0.00040723558f, -0.251846105f },
        { 4.48197079f, 0.0070256046f, -0.292125165f },
        { 3.11746669f, 0.330836475f, 0.0107670771f },
        { 3.44530392f, 0.333118767f, -0.0852379948f },
        { 3.78175545f, 0.361288875f, -0.171207786f },
        { 4.15785265f, 0.365903229f, -0.23702085f },
        { 4.50241852f, 0.327096164f, -0.274928868f },
        { 4.82123899f, 0.34175998f, -0.280468822f },
        { 3.10280824f, 0.675323844f, 0.00882598013f },
        { 3.46457219f, 0.676809847f, -0.0698712245f },
        { 3.81299829f, 0.670834064f, -0.140342325f },
        { 4.14980745f, 0.703446329f, -0.194290563f },
        { 4.49644375f, 0.686310351f, -0.225364491f },
    };
    Triangulation t =
    {
        { 0_v, 1_v, 6_v },
        { 0_v, 6_v, 5_v },
        { 1_v, 2_v, 6_v },
        { 2_v, 7_v, 6_v },
        { 2_v, 3_v, 8_v },
        { 2_v, 8_v, 7_v },
        { 3_v, 4_v, 8_v },
        { 4_v, 9_v, 8_v },
        { 4_v, 10_v, 9_v },
        { 5_v, 6_v, 11_v },
        { 6_v, 12_v, 11_v },
        { 6_v, 7_v, 13_v },
        { 6_v, 13_v, 12_v },
        { 7_v, 8_v, 13_v },
        { 8_v, 14_v, 13_v },
        { 8_v, 9_v, 15_v },
        { 8_v, 15_v, 14_v },
        { 9_v, 10_v, 15_v },
    };
    mesh.topology = MeshBuilder::fromTriangles( t );
    checkSurfaceOffset( mesh, {
        { 3.53456426f, 0.0177574754f, -0.117838085f },
        { 3.58754992f, 0.0137253897f, -0.13228184f },
        { 3.79290915f, 0.0137982592f, -0.185943395f },
        { 3.71790814f, 0.0142281214f, -0.167538032f },
        { 3.74352932f, 0.0159207396f, -0.174424738f },
        { 3.76883101f, 0.0146476077f, -0.181304455f },
        { 3.79533172f, 0.0137018608f, -0.186390594f },
        { 4.15900469f, 0.154283509f, -0.246287376f },
    }, 0.480023875f );
}

} //namespace MR
