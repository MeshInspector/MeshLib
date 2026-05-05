#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRMeshLoad.h>
#include <sstream>

namespace MR
{

// check if Decimator updates region
TEST( MRMesh, MeshDecimate )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettings decimateSettings;
    decimateSettings.maxError = 0.001f;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    auto decimateResults = decimateMesh(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);

    ASSERT_TRUE(meshCylinder.topology.checkValidity());
    meshCylinder.packOptimally();
    ASSERT_TRUE(meshCylinder.topology.checkValidity());
}

TEST( MRMesh, MeshDecimateParallel )
{
    const int cNumVerts = 400;
    auto mesh = makeSphere( { .numMeshVertices = cNumVerts } );
    mesh.packOptimally();
    DecimateSettings settings
    {
        .maxError = 1000000, // no actual limit
        .maxDeletedVertices = cNumVerts - 3,
        .subdivideParts = 8
    };
    decimateMesh( mesh, settings );
    ASSERT_EQ( mesh.topology.numValidFaces(), 2 );
    ASSERT_EQ( mesh.topology.numValidVerts(), 3 );
}

TEST( MRMesh, MeshDecimateMultipleEdgeResolve )
{
    //          2               /
    //         /|\              /
    //       // 5 \\            /
    //     / / / \ \ \          /
    //   /  //     \\  \        /
    //  3---6-------4---1       /
    //    \  \     /  /         /
    //       \ \ / /            /
    //          0               /

    // this test checks scenario where 46 cannot collapse due to creating multiple edge
    // but flipping 26->35 and 24->15 allows the collapse of 46
    VertCoords points( 7 );
    points[0_v] = Vector3f();
    points[1_v] = Vector3f( 3, 2, 0 );
    points[2_v] = Vector3f( 0, 6, 0 );
    points[3_v] = Vector3f( -3, 2, 0 );
    points[4_v] = Vector3f( 0, 2, 0 );
    points[5_v] = Vector3f( 0, 4, 0 );
    points[6_v] = Vector3f( 0, 2, 0 );
    // 46 is degenerate

    Triangulation t( 8 );
    t[0_f] = { 0_v,1_v,4_v };
    t[1_f] = { 0_v,4_v,6_v };
    t[2_f] = { 0_v,6_v,3_v };
    t[3_f] = { 4_v,1_v,2_v };
    t[4_f] = { 4_v,2_v,5_v };
    t[5_f] = { 4_v,5_v,6_v };
    t[6_f] = { 6_v,5_v,2_v };
    t[7_f] = { 6_v,2_v,3_v };
    Mesh mesh = Mesh::fromTriangles( points, t );

    DecimateSettings dsettings
    {
        .strategy = DecimateStrategy::ShortestEdgeFirst,
        .maxEdgeLen = 0.1f,
        .criticalTriAspectRatio = 1e4f,
        .tinyEdgeLength = 0.025f,
        .optimizeVertexPos = false, // this decreases probability of normal inversion near mesh degenerations
        .maxAngleChange = PI_F / 3
    };
    // only allow collapsing 46 edge
    dsettings.preCollapse = [&mesh] ( EdgeId e, const Vector3f )->bool
    {
        auto l = mesh.topology.left( e );
        auto r = mesh.topology.right( e );
        return ( l == 1_f && r == 5_f ) || ( l == 5_f && r == 1_f );
    };
    auto res = decimateMesh( mesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, 1 );

    // rotate triangulation in a way that 46 edge is further from beginning than 26 and 24 (to change decimation queue order)
    std::swap( t[1_f], t[7_f] );
    std::swap( t[5_f], t[6_f] );
    mesh = Mesh::fromTriangles( points, t ); // reversed
    // only allow to collapse 46 edge
    dsettings.preCollapse = [&mesh] ( EdgeId e, const Vector3f )->bool
    {
        auto l = mesh.topology.left( e );
        auto r = mesh.topology.right( e );
        return ( l == 6_f && r == 7_f ) || ( l == 7_f && r == 6_f );
    };
    res = decimateMesh( mesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, 1 );
}

static void testResolveDegen( const char * offMesh, float maxError, int vertsDeleted, int facesDeleted )
{
    std::istringstream s( offMesh );
    auto maybeMesh = MeshLoad::fromOff( s );
    EXPECT_TRUE( maybeMesh );

    DecimateSettings dsettings
    {
        .strategy = DecimateStrategy::ShortestEdgeFirst,
        .maxError = maxError,
        .criticalTriAspectRatio = 1000,
        .tinyEdgeLength = 0.1f * maxError,
        .stabilizer = 1e-6f,
        .optimizeVertexPos = false,
        .maxAngleChange = PI_F / 3
    };

    auto res = decimateMesh( *maybeMesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, vertsDeleted );
    EXPECT_EQ( res.facesDeleted, facesDeleted );
}

TEST( MRMesh, MeshDecimateResolveDegen )
{
    // this case failed before special treatment of pockets in checkDeloneQuadrangle
    testResolveDegen
    (
        "OFF\n"
        "8 8 0\n"
        "\n"
        "0.25991842 0.11936638 0.40698087\n"
        "0.22987929 0.1279805 0.40149236\n"
        "0.25991842 0.11936638 0.4017116\n"
        "0.25991842 0.11936644 0.40171158\n"
        "0.25831458 0.113773495 0.395957\n"
        "0.25991842 0.11936644 0.39124623\n"
        "0.22980908 0.1280007 0.37024236\n"
        "0.25991842 0.11936638 0.37573087\n"
        "\n"
        "3 0 1 2\n"
        "3 3 4 0\n"
        "3 5 2 1\n"
        "3 6 7 5\n"
        "3 7 3 0\n"
        "3 7 4 3\n"
        "3 5 0 2\n"
        "3 0 5 7\n", 8e-6f, 1, 2
    );

    // this case failed before special treatment of pockets in checkDeloneQuadrangle
    testResolveDegen
    (
        "OFF\n"
        "9 8 0\n"
        "\n"
        "0.25991842 0.11936644 0.4636269\n"
        "0.23033604 0.12784958 0.46399236\n"
        "0.25991842 0.11936638 0.45374623\n"
        "0.25991842 0.11936638 0.46948087\n"
        "0.2422457 0.12443432 0.47700337\n"
        "0.25991842 0.11936638 0.4642116\n"
        "0.25991842 0.11936644 0.46421158\n"
        "0.25985792 0.11915526 0.46666718\n"
        "0.25979832 0.118947595 0.4617465\n"
        "\n"
        "3 0 1 2\n"
        "3 3 4 5\n"
        "3 0 5 4\n"
        "3 2 6 3\n"
        "3 7 3 6\n"
        "3 8 6 2\n"
        "3 0 3 5\n"
        "3 2 3 0\n", 8e-6f, 1, 2
    );

    // this case failed before special treatment of MultipleEdge error in MeshDecimate
    testResolveDegen
    (
        "OFF\n"
        "13 18 0\n"
        "\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.36842522 -0.0042799567 -0.0077246446\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.36411813 -0.019299522 -0.023349643\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.3727323 0.010739669 -0.023349643\n"
        "0.36411813 -0.019299552 -0.023349643\n"
        "0.36842522 -0.004279927 -0.013974667\n"
        "0.3598111 -0.034319118 -0.0077246428\n"
        "0.36842522 -0.0042799567 -0.0077246428\n"
        "0.37273225 0.010739639 -0.023349643\n"
        "0.37273225 0.010739639 0.007900357\n"
        "0.36411813 -0.019299552 0.007900357\n"
        "\n"
        "3 0 1 2\n"
        "3 3 2 1\n"
        "3 4 1 5\n"
        "3 1 4 3\n"
        "3 0 5 1\n"
        "3 8 9 6\n"
        "3 10 9 11\n"
        "3 9 12 11\n"
        "3 4 0 7\n"
        "3 0 4 5\n"
        "3 7 0 9\n"
        "3 7 9 10\n"
        "3 9 2 6\n"
        "3 2 9 0\n"
        "3 2 7 6\n"
        "3 7 2 4\n"
        "3 4 2 3\n"
        "3 9 8 12\n", 3e-5f, 6, 12
    );
}

} //namespace MR
