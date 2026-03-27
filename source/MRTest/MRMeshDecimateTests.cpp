#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBuffer.h>

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

} //namespace MR
