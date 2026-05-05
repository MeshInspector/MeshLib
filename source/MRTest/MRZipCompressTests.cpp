#include <MRMesh/MRDirectory.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRMesh/MRZip.h>
#include <MRMesh/MRTimer.h>
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

namespace
{

// Simple LCG used to produce deterministic pseudo-random bytes.
// Keeps the test reproducible across runs and platforms while avoiding
// trivially-compressible input (an all-zeros buffer would make deflate
// look unrealistically good).
inline uint64_t nextLcg( uint64_t & state )
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

} // namespace

// Writes one large file of deterministic pseudo-random bytes to a temp
// folder and compresses that folder to a .zip. End-to-end check of the
// zip write path (libzip + deflate) on a single big near-incompressible
// entry. Pairs with CompressManySmallFilesToZip to contrast one-big-entry
// vs many-small-entry archiver behaviour.
TEST( MRMesh, CompressOneBigFileToZip )
{
    UniqueTemporaryFolder srcFolder;
    ASSERT_TRUE( bool( srcFolder ) );

    // ~120 KB; increase to compress a larger payload.
    constexpr std::size_t fileBytes = 119808;

    const std::filesystem::path filePath = srcFolder / "big.bin";
    {
        uint64_t state = 0x00DDECAFCAFEF00DULL;
        std::vector<char> buf( fileBytes );
        for ( std::size_t j = 0; j < fileBytes; ++j )
            buf[j] = (char)( nextLcg( state ) >> 56 );
        std::ofstream out( filePath, std::ios::binary );
        ASSERT_TRUE( out.is_open() );
        out.write( buf.data(), (std::streamsize)buf.size() );
        ASSERT_TRUE( out.good() );
    }
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::exists( filePath, ec ) );
    const auto fileSize = std::filesystem::file_size( filePath, ec );
    EXPECT_EQ( fileSize, fileBytes );
    spdlog::info( "big.bin size:       {} bytes", fileSize );

    // Zip lands in a separate temp folder so it isn't inside the source tree.
    UniqueTemporaryFolder dstFolder;
    ASSERT_TRUE( bool( dstFolder ) );
    const std::filesystem::path zipPath = dstFolder / "big.zip";

    Timer t( "t" );
    const auto compressRes = compressZip( zipPath, srcFolder );
    const auto sec = t.secondsPassed();

    ASSERT_TRUE( compressRes.has_value() ) << compressRes.error();
    ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) );
    const auto zipSize = std::filesystem::file_size( zipPath, ec );
    EXPECT_GT( zipSize, 0u );
    spdlog::info( "big.zip size:       {} bytes", zipSize );
    spdlog::info( "big.zip compression time: {} sec", sec );

    // Sanity envelope: random input is near-incompressible, so zipSize
    // should be close to fileSize plus a small zip overhead.
    EXPECT_LT( zipSize, fileSize * 2u );
}

// Writes many binary files and same number JSON files to a temporary folder, then
// compresses the folder to a .zip. Pairs with CompressOneBigFileToZip to compare
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

    auto readAllBytes = []( const std::filesystem::path& p ) -> std::vector<char>
    {
        std::ifstream in( p, std::ios::binary | std::ios::ate );
        if ( !in )
            return {};
        const std::streamoff sz = in.tellg();
        in.seekg( 0 );
        std::vector<char> buf( sz < 0 ? 0 : size_t( sz ) );
        if ( !buf.empty() )
            in.read( buf.data(), std::streamsize( buf.size() ) );
        return buf;
    };

    std::error_code ec;
    const std::filesystem::path srcRoot = srcFolder;

    // Exercise every compression level accepted by CompressZipSettings:
    // 0 is documented as "use default level"; 1..9 span fastest -> best-ratio.
    // For each level: compress the source tree, then decompress into a fresh
    // temp folder and verify every file round-trips byte-for-byte. Catches
    // level-dependent wiring bugs (wrong general-purpose bit flag for the
    // entry, mis-encoded compression-method/level byte, store-mode fallthrough
    // on a compressible payload, etc.) that a single-level run would miss.
    for ( int level = 0; level <= 9; ++level )
    {
        UniqueTemporaryFolder dstFolder;
        ASSERT_TRUE( bool( dstFolder ) ) << "level " << level;
        const std::filesystem::path zipPath = dstFolder / "many.zip";

        CompressZipSettings settings;
        settings.compressionLevel = level;

        Timer t( "t" );
        const auto compressRes = compressZip( zipPath, srcFolder, settings );
        const auto sec = t.secondsPassed();
        ASSERT_TRUE( compressRes.has_value() )
            << "level " << level << ": " << compressRes.error();
        ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) ) << "level " << level;
        const auto zipSize = std::filesystem::file_size( zipPath, ec );
        EXPECT_GT( zipSize, 0u ) << "level " << level;
        spdlog::info( "level {}: many.zip size: {} bytes, compression time: {} sec",
            level, zipSize, sec );

        // Sanity envelope: same bound as CompressOneBigFileToZip.
        EXPECT_LT( zipSize, totalInput * 2u ) << "level " << level;

        // Round-trip: decompress into a fresh folder and compare every file's
        // bytes. Built per-iteration so each level gets an independent target
        // tree, with no cross-level contamination.
        UniqueTemporaryFolder roundtripFolder;
        ASSERT_TRUE( bool( roundtripFolder ) ) << "level " << level;
        const auto decompressRes = decompressZip( zipPath, roundtripFolder );
        ASSERT_TRUE( decompressRes.has_value() )
            << "level " << level << ": " << decompressRes.error();

        int verified = 0;
        const std::filesystem::path rtRoot = roundtripFolder;
        for ( auto entry : DirectoryRecursive{ srcRoot, ec } )
        {
            if ( !entry.is_regular_file( ec ) )
                continue;
            const auto rel = std::filesystem::relative( entry.path(), srcRoot, ec );
            const auto dst = rtRoot / rel;
            ASSERT_TRUE( std::filesystem::exists( dst, ec ) )
                << "level " << level << ": missing in roundtrip: " << rel.generic_string();
            const auto origBytes = readAllBytes( entry.path() );
            const auto rtBytes   = readAllBytes( dst );
            ASSERT_EQ( origBytes.size(), rtBytes.size() )
                << "level " << level << ": size mismatch: " << rel.generic_string();
            EXPECT_EQ( origBytes, rtBytes )
                << "level " << level << ": content mismatch: " << rel.generic_string();
            ++verified;
        }
        EXPECT_EQ( verified, numBinaryFiles + numJsonFiles ) << "level " << level;
    }
}

} // namespace MR
