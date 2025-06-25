#include <MRMesh/MRContoursCut.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRRingIterator.h>

namespace MR
{

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
    EXPECT_EQ(   contours[0].size(), 71 );
    EXPECT_EQ(   contours[1].size(), 7 );
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
    EXPECT_EQ(   selfContours[0].size(), 71 );
    EXPECT_TRUE( selfContours[2].size() == 69 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 selfContours[2].size() == 71 ); // with FMA instruction (modern compilers for ARM)
    EXPECT_TRUE( selfContours[4].size() == 9 ||  // without FMA instruction (default settings for x86 or old compilers for ARM)
                 selfContours[4].size() == 7 );  // with FMA instruction (modern compilers for ARM)
    EXPECT_EQ(   selfContours[6].size(), 7 );

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

} //namespace MR
