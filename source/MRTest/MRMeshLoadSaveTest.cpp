#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshLoadObj.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTriMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRImage.h>
#include <MRMesh/MRImageLoad.h>
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

    std::map<std::filesystem::path, std::string> mtlErrors;
    auto res = MeshLoad::fromSceneObjFile( file.data(), file.size(), false, dir, { .mtlErrors = &mtlErrors } );
    std::filesystem::remove( mtlPath );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->size(), 1 );
    const auto& named = res->front();
    EXPECT_EQ( named.mesh.topology.numValidFaces(), 1 );
    ASSERT_EQ( named.textureFiles.size(), 1 );
    EXPECT_EQ( named.textureFiles.front().filename(), "tex1.jpg" );
    ASSERT_TRUE( named.diffuseColor.has_value() );
    EXPECT_TRUE( mtlErrors.empty() );
}

namespace
{

// two closed tetrahedra (no holes, so no warnings about them) with uv-coordinates,
// the first one with material Mat1 and the second one with `material2`, from the given library
std::string twoTetrahedraObj( const std::string& mtlFile, const std::string& material2 = "Mat1" )
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
        "usemtl " + material2 + "\n"
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

// the color of a loaded mesh object, from its materials
Color frontColor( const std::shared_ptr<Object>& obj )
{
    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( obj );
    return objMesh ? objMesh->getFrontColor( false ) : Color();
}

// added after the warnings about missing files
#ifdef __EMSCRIPTEN__
const std::string cWebAdvice = "To load textures in the web app, open a ZIP archive containing the .obj file together with its .mtl and texture files, or use the desktop app.\n";
#else
const std::string cWebAdvice;
#endif

} //anonymous namespace

TEST(MRMesh, LoadObjWithoutMaterials)
{
    const std::string tetrahedron =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 1 4 3\n"
        "f 2 3 4\n";
    UniqueTemporaryFolder dir;
    // no material is lost, so nothing is reported: no library is referenced, even for usemtl,
    // or the missing library is not used, also for "(null)", which Blender writes for the faces without a material
    for ( const std::string prefix : { "", "usemtl default\n", "mtllib model.mtl\n", "mtllib model.mtl\nusemtl (null)\n" } )
    {
        writeTextFile( dir / "model.obj", prefix + tetrahedron );
        auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
        ASSERT_TRUE( res.has_value() );
        EXPECT_EQ( res->objs.size(), 1 );
        EXPECT_EQ( res->warnings, "" );
    }
}

TEST(MRMesh, LoadObjMissingMtl)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );

    std::map<std::filesystem::path, std::string> mtlErrors;
    auto meshes = MeshLoad::fromSceneObjFile( dir / "model.obj", false, { .mtlErrors = &mtlErrors } );
    ASSERT_TRUE( meshes.has_value() );
    EXPECT_EQ( meshes->size(), 2 );
    ASSERT_EQ( mtlErrors.size(), 1 );
    EXPECT_EQ( mtlErrors.begin()->first, dir / "model.mtl" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->objs.size(), 2 );
    // reported once for both objects
    EXPECT_EQ( res->warnings, "Material file model.mtl was not found, so its textures and colors were not loaded.\n" + cWebAdvice );
}

TEST(MRMesh, LoadObjEmptyMtl)
{
    // some exporters write an empty .mtl file
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj",
        "mtllib model.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 1 4 3\n"
        "f 2 3 4\n" );
    writeTextFile( dir / "model.mtl", "" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
}

TEST(MRMesh, LoadObjBrokenMtl)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nKd red\n" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    // no web advice: the library is found
    EXPECT_EQ( res->warnings, "Material file model.mtl could not be loaded (Failed to parse color in MTL-file), so its textures and colors were not loaded.\n" );
}

TEST(MRMesh, LoadObjSeveralMtl)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "a.mtl", "newmtl Mat1\nKd 1 0 0\n" );
    writeTextFile( dir / "b.mtl", "newmtl Mat2\nKd 0 1 0\n" );
    const std::string objs[] = {
        twoTetrahedraObj( "a.mtl b.mtl", "Mat2" ), // several libraries in one line, as the format allows
        twoTetrahedraObj( "a.mtl\nmtllib b.mtl", "Mat2" ), // in consecutive lines
        twoTetrahedraObj( "a.mtl", "Mat2" ) + "mtllib b.mtl\n", // or in separate lines
    };
    for ( const auto& obj : objs )
    {
        writeTextFile( dir / "model.obj", obj );
        auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
        ASSERT_TRUE( res.has_value() );
        EXPECT_EQ( res->warnings, "" );
        ASSERT_EQ( res->objs.size(), 2 );
        EXPECT_EQ( frontColor( res->objs[0] ), Color::red() );
        EXPECT_EQ( frontColor( res->objs[1] ), Color::green() );
    }

    // the materials from the other library are kept
    std::filesystem::remove( dir / "b.mtl" );
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "a.mtl b.mtl", "Mat2" ) );
    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "Material file b.mtl was not found, so its textures and colors were not loaded.\n" + cWebAdvice );
    ASSERT_EQ( res->objs.size(), 2 );
    EXPECT_EQ( frontColor( res->objs[0] ), Color::red() );

    // each library is reported
    std::filesystem::remove( dir / "a.mtl" );
    res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings,
        "Material file a.mtl was not found, so its textures and colors were not loaded.\n"
        "Material file b.mtl was not found, so its textures and colors were not loaded.\n" + cWebAdvice );
}

TEST(MRMesh, LoadObjRepeatedMtl)
{
    // a repeated library takes its last place, since a later library replaces the materials with the same names
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "a.mtl", "newmtl Mat1\nKd 1 0 0\n" );
    writeTextFile( dir / "b.mtl", "newmtl Mat1\nKd 0 1 0\n" );
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "a.mtl\nmtllib b.mtl\nmtllib a.mtl" ) );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    EXPECT_EQ( frontColor( res->objs[0] ), Color::red() );
}

TEST(MRMesh, LoadObjMtlNameWithSpaces)
{
    // MeshSave::toObj names the library after the saved file, e.g. "my model.mtl" for "my model.obj"
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "my model.obj", twoTetrahedraObj( "my model.mtl" ) );

    // named as one file, also if a file named as its last part exists
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nKd 0 1 0\n" );
    auto res = MeshLoad::loadObjectFromObj( dir / "my model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "Material file my model.mtl was not found, so its textures and colors were not loaded.\n" + cWebAdvice );

    writeTextFile( dir / "my model.mtl", "newmtl Mat1\nKd 1 0 0\n" );
    res = MeshLoad::loadObjectFromObj( dir / "my model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    EXPECT_EQ( frontColor( res->objs[0] ), Color::red() );
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

TEST(MRMesh, LoadObjSharedTextures)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    // the first object uses textures a.png and b.png, the second one uses b.png too
    UniqueTemporaryFolder dir;
    auto text = twoTetrahedraObj( "model.mtl", "Mat2" );
    text.insert( text.find( "f 2/2 3/3 4/4\n" ), "usemtl Mat2\n" ); // the last face of the first object
    writeTextFile( dir / "model.obj", text );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nmap_Kd a.png\nnewmtl Mat2\nmap_Kd b.png\n" );
    const Image imageA{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    const Image imageB{ .pixels = { Color::white(), Color::blue(), Color::green(), Color::red() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( imageA, dir / "a.png" ).has_value() );
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( imageB, dir / "b.png" ).has_value() );
    const auto textureA = ImageLoad::fromAnySupportedFormat( dir / "a.png" );
    const auto textureB = ImageLoad::fromAnySupportedFormat( dir / "b.png" );
    ASSERT_TRUE( textureA.has_value() && textureB.has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    auto objMesh1 = std::dynamic_pointer_cast<ObjectMesh>( res->objs[0] );
    auto objMesh2 = std::dynamic_pointer_cast<ObjectMesh>( res->objs[1] );
    ASSERT_TRUE( objMesh1 && objMesh2 );
    const auto& textures1 = objMesh1->getTextures();
    ASSERT_EQ( textures1.size(), 2 );
    EXPECT_EQ( textures1[TextureId( 0 )].pixels, textureA->pixels );
    EXPECT_EQ( textures1[TextureId( 1 )].pixels, textureB->pixels );
    EXPECT_EQ( objMesh1->getTexturePerFace().vec_, ( std::vector<TextureId>{ TextureId( 0 ), TextureId( 0 ), TextureId( 0 ), TextureId( 1 ) } ) );
    const auto& textures2 = objMesh2->getTextures();
    ASSERT_EQ( textures2.size(), 1 );
    EXPECT_EQ( textures2[TextureId( 0 )].pixels, textureB->pixels );
}

TEST(MRMesh, LoadObjMissingTexture)
{
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nmap_Kd texture.png\n" );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    // reported once although both objects use this texture
    EXPECT_EQ( res->warnings, "Texture file texture.png was not found.\n" + cWebAdvice );
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
    EXPECT_EQ( res->warnings, "Texture file texture.txt could not be loaded (" + stringUnsupportedFileExtension() + ").\n" );
}

TEST(MRMesh, LoadObjPartialTextures)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    // one tetrahedron: a face with a missing material, two faces with a texture and a face with a material without one
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj",
        "mtllib model.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vt 1 1\n"
        "usemtl Missing\n"
        "f 1/1 3/3 2/2\n"
        "usemtl Textured\n"
        "f 1/1 2/2 4/4\n"
        "f 1/1 4/4 3/3\n"
        "usemtl Plain\n"
        "f 2/2 3/3 4/4\n" );
    writeTextFile( dir / "model.mtl", "newmtl Textured\nmap_Kd texture.png\nnewmtl Plain\nKd 0 0 1\n" );
    const Image image{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( image, dir / "texture.png" ).has_value() );
    const auto texture = ImageLoad::fromAnySupportedFormat( dir / "texture.png" );
    ASSERT_TRUE( texture.has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" ); // a material missing from a loaded library is not reported
    ASSERT_EQ( res->objs.size(), 1 );
    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( res->objs.front() );
    ASSERT_TRUE( objMesh );

    // the faces without a texture get a transparent one of the same size, so that the object color is shown
    const auto& textures = objMesh->getTextures();
    ASSERT_EQ( textures.size(), 2 );
    EXPECT_EQ( textures[TextureId( 0 )].pixels, std::vector<Color>( 4, Color( 0, 0, 0, 0 ) ) );
    EXPECT_EQ( textures[TextureId( 1 )].pixels, texture->pixels );
    EXPECT_EQ( objMesh->getTexturePerFace().vec_, ( std::vector<TextureId>{ TextureId( 0 ), TextureId( 1 ), TextureId( 1 ), TextureId( 0 ) } ) );
    EXPECT_TRUE( objMesh->getVisualizeProperty( MeshVisualizePropertyType::Texture, ViewportMask::any() ) );

    // MeshLoad::fromObj loads the first texture file
    MeshTexture meshTexture;
    auto mesh = MeshLoad::fromObj( dir / "model.obj", { .texture = &meshTexture } );
    ASSERT_TRUE( mesh.has_value() );
    EXPECT_EQ( meshTexture.pixels, texture->pixels );
}

TEST(MRMesh, LoadObjConsecutiveUsemtl)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    // the last of consecutive usemtl lines is used, and the later material changes of the object are kept
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj",
        "mtllib model.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vt 1 1\n"
        "usemtl Textured\n"
        "usemtl Plain\n"
        "f 1/1 3/3 2/2\n"
        "f 1/1 2/2 4/4\n"
        "usemtl Textured\n"
        "f 1/1 4/4 3/3\n"
        "usemtl Plain\n"
        "f 2/2 3/3 4/4\n" );
    // both materials have a color, since a material without one resets the object color as contradicting
    writeTextFile( dir / "model.mtl", "newmtl Textured\nKd 0 0 1\nmap_Kd texture.png\nnewmtl Plain\nKd 0 0 1\n" );
    const Image image{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( image, dir / "texture.png" ).has_value() );
    const auto texture = ImageLoad::fromAnySupportedFormat( dir / "texture.png" );
    ASSERT_TRUE( texture.has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 1 );
    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( res->objs.front() );
    ASSERT_TRUE( objMesh );
    const auto& textures = objMesh->getTextures();
    ASSERT_EQ( textures.size(), 2 );
    EXPECT_EQ( textures[TextureId( 0 )].pixels, std::vector<Color>( 4, Color( 0, 0, 0, 0 ) ) );
    EXPECT_EQ( textures[TextureId( 1 )].pixels, texture->pixels );
    EXPECT_EQ( objMesh->getTexturePerFace().vec_, ( std::vector<TextureId>{ TextureId( 0 ), TextureId( 0 ), TextureId( 1 ), TextureId( 0 ) } ) );
    EXPECT_EQ( objMesh->getFrontColor( false ), Color::blue() );
}

TEST(MRMesh, LoadObjFacesBeforeFirstMaterial)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    // the faces before the first usemtl line have no material, not the first material of the file
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj",
        "mtllib model.mtl\n"
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
        "usemtl Textured\n"
        "f 1/1 4/4 3/3\n"
        "f 2/2 3/3 4/4\n" );
    writeTextFile( dir / "model.mtl", "newmtl Textured\nKd 0 0 1\nmap_Kd texture.png\n" );
    const Image image{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( image, dir / "texture.png" ).has_value() );
    const auto texture = ImageLoad::fromAnySupportedFormat( dir / "texture.png" );
    ASSERT_TRUE( texture.has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 1 );
    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( res->objs.front() );
    ASSERT_TRUE( objMesh );
    // the faces without a material get the transparent texture and don't change the object color
    const auto& textures = objMesh->getTextures();
    ASSERT_EQ( textures.size(), 2 );
    EXPECT_EQ( textures[TextureId( 0 )].pixels, std::vector<Color>( 4, Color( 0, 0, 0, 0 ) ) );
    EXPECT_EQ( textures[TextureId( 1 )].pixels, texture->pixels );
    EXPECT_EQ( objMesh->getTexturePerFace().vec_, ( std::vector<TextureId>{ TextureId( 0 ), TextureId( 0 ), TextureId( 1 ), TextureId( 1 ) } ) );
    EXPECT_EQ( objMesh->getFrontColor( false ), Color::blue() );

    // MeshLoad::fromObj skips the transparent texture and loads the texture file
    MeshTexture meshTexture;
    auto mesh = MeshLoad::fromObj( dir / "model.obj", { .texture = &meshTexture } );
    ASSERT_TRUE( mesh.has_value() );
    EXPECT_EQ( meshTexture.pixels, texture->pixels );
}

TEST(MRMesh, LoadObjUnusedMaterialName)
{
    // b.mtl is missing and a.mtl has no Mat2, but no face uses Mat2: no material is lost, so nothing is reported
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "a.mtl", "newmtl Mat1\nKd 1 0 0\n" );
    auto replaced = twoTetrahedraObj( "a.mtl b.mtl" );
    replaced.insert( replaced.find( "usemtl Mat1\n" ), "usemtl Mat2\n" );
    const std::string objs[] = {
        replaced, // followed by another usemtl line
        twoTetrahedraObj( "a.mtl b.mtl" ) + "usemtl Mat2\n", // after the last face
    };
    for ( const auto& obj : objs )
    {
        writeTextFile( dir / "model.obj", obj );
        auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
        ASSERT_TRUE( res.has_value() );
        EXPECT_EQ( res->warnings, "" );
        ASSERT_EQ( res->objs.size(), 2 );
        EXPECT_EQ( frontColor( res->objs[0] ), Color::red() );
        EXPECT_EQ( frontColor( res->objs[1] ), Color::red() );
    }
}

TEST(MRMesh, LoadObjMaterialPerObject)
{
    if ( !ImageSave::getImageSaver( "*.png" ) || !ImageLoad::getImageLoader( "*.png" ) )
    {
        GTEST_SKIP() << "PNG format is not supported in this build";
    }

    // usemtl of the second object comes right before its faces, as usual: only its own material is used
    UniqueTemporaryFolder dir;
    writeTextFile( dir / "model.obj", twoTetrahedraObj( "model.mtl", "Mat2" ) );
    writeTextFile( dir / "model.mtl", "newmtl Mat1\nKd 0 0 1\nnewmtl Mat2\nKd 1 0 0\nmap_Kd texture.png\n" );
    const Image image{ .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() }, .resolution = { 2, 2 } };
    ASSERT_TRUE( ImageSave::toAnySupportedFormat( image, dir / "texture.png" ).has_value() );

    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->warnings, "" );
    ASSERT_EQ( res->objs.size(), 2 );
    auto objMesh1 = std::dynamic_pointer_cast<ObjectMesh>( res->objs[0] );
    auto objMesh2 = std::dynamic_pointer_cast<ObjectMesh>( res->objs[1] );
    ASSERT_TRUE( objMesh1 && objMesh2 );
    EXPECT_EQ( objMesh1->getFrontColor( false ), Color::blue() );
    EXPECT_TRUE( objMesh1->getTextures().empty() );
    EXPECT_EQ( objMesh2->getFrontColor( false ), Color::red() );
    EXPECT_EQ( objMesh2->getTextures().size(), 1 );
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

namespace
{

// a closed tetrahedron, whose vertices follow the first `numPrevVerts` vertices of the file
std::string tetrahedronObj( int numPrevVerts )
{
    const auto face = [numPrevVerts] ( int a, int b, int c )
    {
        return "f " + std::to_string( numPrevVerts + a ) + " " + std::to_string( numPrevVerts + b ) + " " + std::to_string( numPrevVerts + c ) + "\n";
    };
    return "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n" + face( 1, 3, 2 ) + face( 1, 2, 4 ) + face( 1, 4, 3 ) + face( 2, 3, 4 );
}

using NamedFaceCounts = std::vector<std::pair<std::string, int>>;

// the name and the number of faces of each loaded mesh
NamedFaceCounts namedFaceCounts( const std::vector<MeshLoad::NamedMesh>& meshes )
{
    NamedFaceCounts res;
    for ( const auto& m : meshes )
        res.emplace_back( m.name, m.mesh.topology.numValidFaces() );
    return res;
}

} //anonymous namespace

TEST(MRMesh, LoadObjFacesBeforeFirstObject)
{
    // the faces before the first o line form an object without a name
    const std::pair<std::string, NamedFaceCounts> cases[] = {
        { tetrahedronObj( 0 ) + "o A\n" + tetrahedronObj( 4 ) + "o B\n" + tetrahedronObj( 8 ), { { "", 4 }, { "A", 4 }, { "B", 4 } } },
        { tetrahedronObj( 0 ) + "o A\n" + tetrahedronObj( 4 ), { { "", 4 }, { "A", 4 } } }, // also with a single o line
    };
    UniqueTemporaryFolder dir;
    for ( const auto& [obj, expected] : cases )
    {
        writeTextFile( dir / "model.obj", obj );
        auto res = MeshLoad::fromSceneObjFile( dir / "model.obj", false );
        ASSERT_TRUE( res.has_value() ) << res.error();
        EXPECT_EQ( namedFaceCounts( *res ), expected );
    }

    // which gets the name of the file
    writeTextFile( dir / "model.obj", tetrahedronObj( 0 ) + "o A\n" + tetrahedronObj( 4 ) );
    auto res = MeshLoad::loadObjectFromObj( dir / "model.obj" );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->objs.size(), 2 );
    EXPECT_EQ( res->objs[0]->name(), "model" );
    EXPECT_EQ( res->objs[1]->name(), "A" );
}

TEST(MRMesh, LoadObjObjectWithoutFaces)
{
    // an object without faces, e.g. with only lines, is skipped wherever it is
    const std::pair<std::string, NamedFaceCounts> cases[] = {
        { "o A\n" + tetrahedronObj( 0 ) + "o Lines\nl 1 2 3 4\no B\n" + tetrahedronObj( 4 ), { { "A", 4 }, { "B", 4 } } },
        { "o A\n" + tetrahedronObj( 0 ) + "o B\n" + tetrahedronObj( 4 ) + "o Lines\nl 1 2 3 4\n", { { "A", 4 }, { "B", 4 } } },
        { "o Lines\nl 1 2 3 4\no A\n" + tetrahedronObj( 0 ), { { "A", 4 } } },
        { "o A\n" + tetrahedronObj( 0 ) + "o Lines\nl 1 2 3 4\n", { { "A", 4 } } },
    };
    UniqueTemporaryFolder dir;
    for ( const auto& [obj, expected] : cases )
    {
        writeTextFile( dir / "model.obj", obj );
        auto res = MeshLoad::fromSceneObjFile( dir / "model.obj", false );
        ASSERT_TRUE( res.has_value() ) << res.error();
        EXPECT_EQ( namedFaceCounts( *res ), expected );
    }
}

TEST(MRMesh, LoadObjConsecutiveObjectNames)
{
    // of consecutive o lines, the last one names the faces after them
    const std::pair<std::string, NamedFaceCounts> cases[] = {
        { "o Empty\no A\n" + tetrahedronObj( 0 ), { { "A", 4 } } },
        { "o Empty\no A\n" + tetrahedronObj( 0 ) + "o Empty\no B\n" + tetrahedronObj( 4 ), { { "A", 4 }, { "B", 4 } } },
    };
    UniqueTemporaryFolder dir;
    for ( const auto& [obj, expected] : cases )
    {
        writeTextFile( dir / "model.obj", obj );
        auto res = MeshLoad::fromSceneObjFile( dir / "model.obj", false );
        ASSERT_TRUE( res.has_value() ) << res.error();
        EXPECT_EQ( namedFaceCounts( *res ), expected );
    }
}

} //namespace MR
