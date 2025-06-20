#include <MRMesh/MRContoursCut.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>

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

    const auto conv = getVectorConverters( meshA, meshB );

    const auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, conv.toInt );
    EXPECT_EQ( intersections.size(), 152 );

    const auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    EXPECT_EQ( contours.size(), 4 );
    // FIXME: the results are platform-dependent
    EXPECT_EQ(   contours[0].size(), 71 );
    EXPECT_EQ(   contours[1].size(), 7 );
    EXPECT_TRUE( contours[2].size() == 69 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 contours[2].size() == 71 ); // with FMA instruction (modern compilers for ARM)
    EXPECT_TRUE( contours[3].size() == 9 ||  // without FMA instruction (default settings for x86 or old compilers for ARM)
                 contours[3].size() == 7 );  // with FMA instruction (modern compilers for ARM)

    OneMeshContours meshAContours, meshBContours;
    getOneMeshIntersectionContours( meshA, meshB, contours, &meshAContours, &meshBContours, conv );
    EXPECT_EQ( meshAContours.size(), 4 );
    EXPECT_EQ( meshBContours.size(), 4 );

    size_t posCount = 0;
    for ( const auto& contour : meshAContours )
        posCount += contour.intersections.size();
    EXPECT_EQ( posCount, 156 );
}

} //namespace MR
