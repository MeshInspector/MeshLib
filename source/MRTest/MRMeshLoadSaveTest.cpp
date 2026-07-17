#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshLoadObj.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTriMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRColor.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

namespace MR
{

TEST(MRMesh, LoadSave) 
{
    std::string file = 
        "OFF\n"
        "5 6 0\n"

        "0 0 1\n"
        "1 0 0\n"
        "0 1 0\n"
        "-1 0 0\n"
        "0 -1 0\n"

        "3 0 1 2\n"
        "3 0 2 3\n"
        "3 0 3 4\n"
        "3 0 4 1\n"
        "3 1 3 2\n"
        "3 1 4 3\n";
     
    std::istringstream in( file );

    auto loadRes = MeshLoad::fromOff( in );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );

    auto box = loadRes->computeBoundingBox();
    EXPECT_EQ( box, Box3f( Vector3f(-1, -1, 0), Vector3f(1, 1, 1) ) );
    EXPECT_TRUE ( box.contains( Vector3f(0, 0, 0) ) );
    EXPECT_FALSE( box.contains( Vector3f(-1, -1, -1) ) );

    std::stringstream ss;
    auto saveRes = MeshSave::toOff( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromOff( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
    
    // save/load to internal format
    ss = std::stringstream{};
    saveRes = MeshSave::toMrmesh( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromMrmesh( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );

    // save/load to binary STL format
    ss = std::stringstream{};
    saveRes = MeshSave::toBinaryStl( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromBinaryStl( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
}

TEST(MRMesh, TriMeshSavePly)
{
    TriMesh triMesh;
    triMesh.tris = Triangulation{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v },
        { 0_v, 3_v, 4_v },
        { 0_v, 4_v, 1_v },
        { 1_v, 3_v, 2_v },
        { 1_v, 4_v, 3_v }
    };
    triMesh.points.emplace_back( 0.f, 0.f, 1.f );
    triMesh.points.emplace_back( 1.f, 0.f, 0.f );
    triMesh.points.emplace_back( 0.f, 1.f, 0.f );
    triMesh.points.emplace_back( -1.f, 0.f, 0.f );
    triMesh.points.emplace_back( 0.f, -1.f, 0.f );

    std::ostringstream outTriMesh;
    EXPECT_TRUE( MeshSave::toPly( triMesh, outTriMesh ).has_value() );

    // the same bytes must be saved for TriMesh and for equivalent Mesh
    const auto mesh = Mesh::fromTriMesh( TriMesh( triMesh ) );
    std::ostringstream outMesh;
    EXPECT_TRUE( MeshSave::toPly( mesh, outMesh ).has_value() );
    EXPECT_EQ( outTriMesh.str(), outMesh.str() );

    std::istringstream in( outTriMesh.str() );
    auto loadRes = MeshLoad::fromPly( in );
    ASSERT_TRUE( loadRes.has_value() );
    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
}

TEST(MRMesh, LoadPlyDuplicatingNonManifoldVertices)
{
    // two closed triangle fans sharing only the central vertex #0
    TriMesh triMesh;
    triMesh.tris = Triangulation{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v },
        { 0_v, 3_v, 1_v },
        { 0_v, 4_v, 5_v },
        { 0_v, 5_v, 6_v },
        { 0_v, 6_v, 4_v }
    };
    triMesh.points.emplace_back( 0.f, 0.f, 0.f );
    triMesh.points.emplace_back( 1.f, 0.f, 0.f );
    triMesh.points.emplace_back( 0.f, 1.f, 0.f );
    triMesh.points.emplace_back( 0.f, 0.f, 1.f );
    triMesh.points.emplace_back( -1.f, 0.f, 0.f );
    triMesh.points.emplace_back( 0.f, -1.f, 0.f );
    triMesh.points.emplace_back( 0.f, 0.f, -1.f );

    VertColors colors;
    for ( int i = 0; i < 7; ++i )
        colors.push_back( Color( i, 0, 0 ) );

    std::ostringstream out;
    ASSERT_TRUE( MeshSave::toPly( triMesh, out, { .colors = &colors } ).has_value() );

    VertColors loadedColors;
    int dupCount = 0;
    std::istringstream in( out.str() );
    auto loadRes = MeshLoad::fromPly( in, { .colors = &loadedColors, .duplicatedVertexCount = &dupCount } );
    ASSERT_TRUE( loadRes.has_value() );
    EXPECT_EQ( dupCount, 1 );
    EXPECT_EQ( loadRes->points.size(), 8 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 8 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
    EXPECT_EQ( loadRes->points[7_v], triMesh.points[0_v] );
    ASSERT_EQ( loadedColors.size(), 8 );
    EXPECT_EQ( loadedColors[7_v], colors[0_v] );
}

TEST(MRMesh, LoadPlyPointCloud)
{
    // PLY point cloud with a face element having zero faces must keep all its points
    std::string file =
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n"
        "1 0 0\n";

    std::istringstream in( file );
    auto loadRes = MeshLoad::fromPly( in );
    ASSERT_TRUE( loadRes.has_value() );
    EXPECT_EQ( loadRes->points.size(), 2 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 0 );
}

TEST(MRMesh, LoadAsciiStlTooFewVertices)
{
    std::string file =
        "solid Name\n"
        "facet normal 0 0 1\n"
        "outer loop\n"
        "vertex 0 0 0\n"
        "vertex 1 0 0\n"
        "endloop\n"
        "endfacet\n"
        "endsolid Name\n";

    std::istringstream in( file );
    auto loadRes = MeshLoad::fromASCIIStl( in );
    EXPECT_FALSE( loadRes.has_value() );
}

TEST(MRMesh, LoadObjTabIndented)
{
    // some exporters (e.g. 3ds Max guruware OBJ exporter) indent lines with tabs
    const auto dir = std::filesystem::temp_directory_path();
    const auto mtlPath = dir / "MRLoadObjTabIndented.mtl";
    {
        std::ofstream mtl( mtlPath, std::ios::binary );
        mtl <<
            "newmtl Mat1\n"
            "\tKd 0.5880 0.5880 0.5880\n"
            "\tmap_Kd tex1.jpg\n";
    }

    std::string file =
        "mtllib MRLoadObjTabIndented.mtl\n"
        "v 0 0 0\n"
        "\tv 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "\tusemtl Mat1\n"
        "f 1/1 2/2 3/3\n";

    auto res = MeshLoad::fromSceneObjFile( file.data(), file.size(), false, dir );
    std::filesystem::remove( mtlPath );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->size(), 1 );
    const auto& named = res->front();
    EXPECT_EQ( named.mesh.topology.numValidFaces(), 1 );
    ASSERT_EQ( named.textureFiles.size(), 1 );
    EXPECT_EQ( named.textureFiles.front().filename(), "tex1.jpg" );
    ASSERT_TRUE( named.diffuseColor.has_value() );
}

} //namespace MR
