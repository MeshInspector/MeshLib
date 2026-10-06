#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshLoadObj.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTriMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRImage.h>
#include <MRMesh/MRImageSave.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
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

TEST(MRMesh, StlLoadAsTriMesh)
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
    const auto mesh = Mesh::fromTriMesh( TriMesh( triMesh ) );

    std::stringstream binStream;
    EXPECT_TRUE( MeshSave::toBinaryStl( mesh, binStream ).has_value() );
    auto binTriMesh = loadBinaryStlAsTriMesh( binStream );
    ASSERT_TRUE( binTriMesh.has_value() );
    EXPECT_EQ( binTriMesh->tris.size(), 6 );
    EXPECT_EQ( binTriMesh->points.size(), 5 );

    // mesh made from loaded TriMesh must be equal to directly loaded mesh
    binStream.clear();
    binStream.seekg( 0 );
    auto binMesh = loadBinaryStl( binStream );
    ASSERT_TRUE( binMesh.has_value() );
    EXPECT_EQ( Mesh::fromTriMesh( std::move( *binTriMesh ) ), *binMesh );

    std::stringstream ascStream;
    EXPECT_TRUE( MeshSave::toAsciiStl( mesh, ascStream ).has_value() );
    auto ascTriMesh = loadASCIIStlAsTriMesh( ascStream );
    ASSERT_TRUE( ascTriMesh.has_value() );
    EXPECT_EQ( ascTriMesh->tris.size(), 6 );
    EXPECT_EQ( ascTriMesh->points.size(), 5 );

    ascStream.clear();
    ascStream.seekg( 0 );
    auto ascMesh = loadASCIIStl( ascStream );
    ASSERT_TRUE( ascMesh.has_value() );
    EXPECT_EQ( Mesh::fromTriMesh( std::move( *ascTriMesh ) ), *ascMesh );

    const auto file = std::filesystem::temp_directory_path() / "MRStlLoadAsTriMesh.stl";
    EXPECT_TRUE( MeshSave::toAsciiStl( mesh, file ).has_value() );
    auto fileTriMesh = loadASCIIStlAsTriMesh( file );
    std::filesystem::remove( file );
    ASSERT_TRUE( fileTriMesh.has_value() );
    EXPECT_EQ( fileTriMesh->tris.size(), 6 );
    EXPECT_EQ( fileTriMesh->points.size(), 5 );
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
    EXPECT_TRUE( named.mtlError.empty() );
}

namespace
{

// two closed tetrahedra (no holes, so no warnings about them) with uv-coordinates and material Mat1 from the given library
std::string twoTetrahedraObj( const std::string& mtlFile )
{
    return
        "mtllib " + mtlFile + "\n"
        "usemtl Mat1\n"
        "o Tetrahedron1\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vt 1 1\n"
        "f 1/1 3/3 2/2\n"
        "f 1/1 2/2 4/4\n"
        "f 1/1 4/4 3/3\n"
        "f 2/2 3/3 4/4\n"
        "o Tetrahedron2\n"
        "v 2 0 0\n"
        "v 3 0 0\n"
        "v 2 1 0\n"
        "v 2 0 1\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vt 1 1\n"
        "f 5/5 7/7 6/6\n"
        "f 5/5 6/6 8/8\n"
        "f 5/5 8/8 7/7\n"
        "f 6/6 7/7 8/8\n";
}

void writeTextFile( const std::filesystem::path& path, const std::string& text )
{
    std::ofstream out( path, std::ios::binary );
    out << text;
}

// added after the warnings about missing files
#ifdef __EMSCRIPTEN__
const std::string cWebAdvice = "To load textures in the web app, open a ZIP archive containing the .obj file together with its .mtl and texture files, or use the desktop app.\n";
#else
const std::string cWebAdvice;
#endif

} //anonymous namespace

TEST(MRMesh, LoadObjMissingMtl)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );

    auto meshes = MeshLoad::fromSceneObjFile( dir / "model.obj", false );
    ASSERT_TRUE( meshes.has_value() );
    ASSERT_EQ( meshes->size(), 2 );
    for ( const auto& m : *meshes )
    {
        EXPECT_EQ( m.mtlError, "Material file model.mtl was not found" );
    }

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->objs.size(), 2 );
    // reported once for both objects
    EXPECT_EQ( res->warnings, "Material file model.mtl was not found, so textures and material colors were not loaded.\n" + cWebAdvice );
}

TEST(MRMesh, LoadObjWithTexture)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nKd 1 0 0\nmap_Kd texture.png\n" );
    const Image texture{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( texture, dir / "texture.png" ).has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    for ( const auto& obj : res->objs )
    {
        auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( obj );
        ASSERT_TRUE( objMesh );
        EXPECT_EQ( objMesh->getTextures().size(), 1 );
        EXPECT_EQ( objMesh->getFrontColor( false ), Color::red() );
    }
}

TEST(MRMesh, LoadObjMissingTexture)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nmap_Kd texture.png\n" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    // reported once although both objects use this texture
    EXPECT_EQ( res->warnings, "Texture file texture.png was not found, so textures were not loaded.\n" + cWebAdvice );
    ASSERT_EQ( res->objs.size(), 2 );
    for ( const auto& obj : res->objs )
    {
        auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( obj );
        ASSERT_TRUE( objMesh );
        EXPECT_TRUE( objMesh->getTextures().empty() );
    }
}

TEST(MRMesh, LoadObjUnsupportedTexture)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nmap_Kd texture.txt\n" );
    writeTextFile( dir / "texture.txt", "not an image" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    // no web advice: opening the files from a ZIP archive would not help
    EXPECT_EQ( res->warnings, "Texture file texture.txt could not be loaded (" + stringUnsupportedFileExtension() + "), so textures were not loaded.\n" );
}

TEST(MRMesh, LoadObjUtf8MtlName)
{
    const std::string mtlFile = "\xd0\xbc\xd0\xbe\xd0\xb4\xd0\xb5\xd0\xbb\xd1\x8c.mtl"; // non-ASCII name in UTF-8
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( mtlFile ) );
    writeTextFile( dir / asU8String( mtlFile ), "newmtl Mat1\nKd 1 0 0\n" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( res->objs.front() );
    ASSERT_TRUE( objMesh );
    EXPECT_EQ( objMesh->getFrontColor( false ), Color::red() );
}

} //namespace MR
