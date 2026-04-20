#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRMesh/MRZip.h>
#include <MRPch/MRSpdlog.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace MR
{

<<<<<<< HEAD
// Writes a sphere to a .mrmesh file in a temporary folder, then
=======
// Writes a ~100K-vertex sphere to a .mrmesh file in a temporary folder, then
>>>>>>> 6dadccc9 (test: add sphere mesh compress-to-zip test)
// compresses that folder to a .zip and verifies the archive was created and
// is non-empty. Serves as a realistic end-to-end exercise of MeshLib's zip
// write path (libzip + deflate) on mesh-sized data.
TEST( MRMesh, CompressSphereToZip )
{
    // No-op re-trigger marker for CI; remove in cleanup commit.
    UniqueTemporaryFolder srcFolder;
    ASSERT_TRUE( bool( srcFolder ) );

    // Generate a sphere with ~100K vertices. makeSphere's subdivision
    // targets the requested count but may land a handful over.
    constexpr int targetVerts = 100'000;
    SphereParams params;
    params.radius = 1.0f;
    params.numMeshVertices = targetVerts;
    const Mesh sphere = makeSphere( params );
    EXPECT_EQ( (int)sphere.topology.numValidVerts(), targetVerts );

    // Save mesh as a .mrmesh file in the temp folder.
    const std::filesystem::path meshPath = srcFolder / "sphere.mrmesh";
    const auto saveRes = MeshSave::toMrmesh( sphere, meshPath );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
<<<<<<< HEAD
<<<<<<< HEAD
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::exists( meshPath, ec ) );
    const auto meshSize = std::filesystem::file_size( meshPath, ec );
    EXPECT_GT( meshSize, 0u );
    spdlog::info( "sphere.mrmesh size: {} bytes", meshSize );
<<<<<<< HEAD
=======
    ASSERT_TRUE( std::filesystem::exists( meshPath ) );
    const auto meshSize = std::filesystem::file_size( meshPath );
=======
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::exists( meshPath, ec ) );
    const auto meshSize = std::filesystem::file_size( meshPath, ec );
>>>>>>> 1cb6fcc9 (fix)
    EXPECT_GT( meshSize, 0u );
>>>>>>> 6dadccc9 (test: add sphere mesh compress-to-zip test)
=======
>>>>>>> a700f5ed (test: log mesh and zip sizes via spdlog::info)

    // Compress the temp folder into a .zip located in a second temp folder
    // (so the zip isn't inside the folder being compressed).
    UniqueTemporaryFolder dstFolder;
    ASSERT_TRUE( bool( dstFolder ) );
    const std::filesystem::path zipPath = dstFolder / "sphere.zip";

    const auto compressRes = compressZip( zipPath, srcFolder );
    ASSERT_TRUE( compressRes.has_value() ) << compressRes.error();
<<<<<<< HEAD
<<<<<<< HEAD
    ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) );
    const auto zipSize = std::filesystem::file_size( zipPath, ec );
    EXPECT_GT( zipSize, 0u );
    spdlog::info( "sphere.zip size:    {} bytes", zipSize );
<<<<<<< HEAD
=======
    ASSERT_TRUE( std::filesystem::exists( zipPath ) );
    const auto zipSize = std::filesystem::file_size( zipPath );
=======
    ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) );
    const auto zipSize = std::filesystem::file_size( zipPath, ec );
>>>>>>> 1cb6fcc9 (fix)
    EXPECT_GT( zipSize, 0u );
>>>>>>> 6dadccc9 (test: add sphere mesh compress-to-zip test)
=======
>>>>>>> a700f5ed (test: log mesh and zip sizes via spdlog::info)

    // Sanity: the zip should not be absurdly larger than the source
    // (that would indicate something is wrong with the envelope); and
    // since .mrmesh is a raw binary dump of topology plus coordinate
    // floats, deflate typically produces a modestly smaller archive.
    EXPECT_LT( zipSize, meshSize * 2u );
}

<<<<<<< HEAD
// Writes many binary files and same number JSON files to a temporary folder, then
// compresses the folder to a .zip. Pairs with CompressSphereToZip to compare
// compression of one large binary vs many small mixed-type entries.
//
// libzip compresses each entry independently, so per-entry overhead (local
// file header, CRC32 pass, separate deflate session) can dominate when the
// archive is made of many small files. This test makes that cost visible.
TEST( MRMesh, CompressManySmallFilesToZip )
{
    UniqueTemporaryFolder srcFolder;
    ASSERT_TRUE( bool( srcFolder ) );

    // increase both below numbers to make the files being compressed larger, 200 * 2 files * 60'000 bytes -> 24M bytes
    constexpr int numBinaryFiles = 20;
    constexpr int numJsonFiles = numBinaryFiles;
    constexpr size_t bytesPerFile = 6000;

    // Simple LCG used to produce deterministic pseudo-random bytes.
    // Keeps the test reproducible across runs and platforms while avoiding
    // trivially-compressible input (an all-zeros buffer would make deflate
    // look unrealistically good).
    auto nextLcg = []( uint64_t & state ) -> uint64_t
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    };

    auto makeName = []( const char * prefix, int i, const char * ext )
    {
        char buf[64];
        std::snprintf( buf, sizeof( buf ), "%s_%03d.%s", prefix, i, ext );
        return std::string( buf );
    };

    // Binary files of pseudo-random bytes. Poor compressibility on
    // purpose — representative of mesh coordinate floats, compressed-texture
    // blobs, and other near-incompressible payloads that often live in a
    // MeshLib scene save.
    std::size_t totalBinaryBytes = 0;
    std::vector<char> binBuf( bytesPerFile );
    for ( int i = 0; i < numBinaryFiles; ++i )
    {
        uint64_t state = 0x1234567890ABCDEFULL ^ ( (uint64_t)i << 1 );
        for ( size_t j = 0; j < bytesPerFile; ++j )
            binBuf[j] = (char)( nextLcg( state ) >> 56 );

        const std::filesystem::path p = srcFolder / makeName( "data", i, "bin" );
        std::ofstream out( p, std::ios::binary );
        ASSERT_TRUE( out.is_open() );
        out.write( binBuf.data(), (std::streamsize)binBuf.size() );
        ASSERT_TRUE( out.good() );
        out.close();
        totalBinaryBytes += bytesPerFile;
    }

    // JSON files of deterministic structured-looking text. Highly
    // compressible — representative of scene-description metadata, logs,
    // shader source, and other textual payloads.
    std::size_t totalJsonBytes = 0;
    for ( int i = 0; i < numJsonFiles; ++i )
    {
        uint64_t state = 0xDEADBEEFCAFEBABEULL ^ ( (uint64_t)i << 1 );

        std::string text;
        text.reserve( bytesPerFile + 256 );
        text += "[\n";
        int idx = 0;
        while ( text.size() + 96 < bytesPerFile )
        {
            if ( idx > 0 )
                text += ",\n";
            const uint32_t rx = (uint32_t)( nextLcg( state ) >> 32 );
            const uint32_t ry = (uint32_t)( nextLcg( state ) >> 32 );
            const uint32_t rz = (uint32_t)( nextLcg( state ) >> 32 );
            char line[128];
            const int n = std::snprintf( line, sizeof( line ),
                "  {\"id\": %d, \"x\": %.6f, \"y\": %.6f, \"z\": %.6f}",
                idx,
                (double)rx / 4294967296.0,
                (double)ry / 4294967296.0,
                (double)rz / 4294967296.0 );
            ASSERT_GT( n, 0 );
            text.append( line, (size_t)n );
            ++idx;
        }
        text += "\n]\n";
        // Pad to exactly bytesPerFile with trailing spaces so the per-file
        // size — and therefore the total — is deterministic across runs.
        // The file is never parsed, so trailing whitespace past the final
        // ']' is harmless.
        if ( text.size() < bytesPerFile )
            text.append( bytesPerFile - text.size(), ' ' );
        else if ( text.size() > bytesPerFile )
            text.resize( bytesPerFile );

        const std::filesystem::path p = srcFolder / makeName( "meta", i, "json" );
        std::ofstream out( p, std::ios::binary );
        ASSERT_TRUE( out.is_open() );
        out.write( text.data(), (std::streamsize)text.size() );
        ASSERT_TRUE( out.good() );
        out.close();
        totalJsonBytes += text.size();
    }

    const std::size_t totalInput = totalBinaryBytes + totalJsonBytes;
    spdlog::info( "many-files input:   {} binary + {} json = {} bytes",
        totalBinaryBytes, totalJsonBytes, totalInput );

    // Compress to a zip in a separate temp folder.
    UniqueTemporaryFolder dstFolder;
    ASSERT_TRUE( bool( dstFolder ) );
    const std::filesystem::path zipPath = dstFolder / "many.zip";

    const auto compressRes = compressZip( zipPath, srcFolder );
    ASSERT_TRUE( compressRes.has_value() ) << compressRes.error();
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) );
    const auto zipSize = std::filesystem::file_size( zipPath, ec );
    EXPECT_GT( zipSize, 0u );
    spdlog::info( "many.zip size:      {} bytes", zipSize );

    // Sanity envelope: same bound as the sphere test.
    EXPECT_LT( zipSize, totalInput * 2u );
}

=======
>>>>>>> 6dadccc9 (test: add sphere mesh compress-to-zip test)
} // namespace MR
