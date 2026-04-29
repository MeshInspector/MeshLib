#include <MRMesh/MRGTest.h>
#include <MRMesh/MRZlib.h>

#include <cstdint>

namespace
{

/// independent CRC-32 reference (PKZIP polynomial 0xedb88320, init 0xffffffff, final xor 0xffffffff)
uint32_t crc32Ref( const unsigned char* data, size_t len )
{
    uint32_t crc = 0xffffffffu;
    for ( size_t i = 0; i < len; ++i )
    {
        crc ^= data[i];
        for ( int k = 0; k < 8; ++k )
            crc = ( crc & 1u ) ? ( crc >> 1 ) ^ 0xedb88320u : ( crc >> 1 );
    }
    return ~crc;
}

constexpr unsigned char cInput[] = {
    0xe1, 0x83, 0x9b, 0xe1, 0x83, 0x94, 0xe1, 0x83, 0xaa, 0xe1, 0x83, 0xae,
    0xe1, 0x83, 0x90, 0xe1, 0x83, 0xa2, 0xe1, 0x83, 0x94, 0xe1, 0x83, 0x9b,
    0x20, 0xe1, 0x83, 0xa9, 0xe1, 0x83, 0x90, 0xe1, 0x83, 0xac, 0xe1, 0x83,
    0xa7, 0xe1, 0x83, 0x90, 0xe1, 0x83, 0x9c, 0xe1, 0x83, 0xa3, 0xe1, 0x83,
    0xa0, 0xe1, 0x83, 0x98, 0x20, 0xe1, 0x83, 0xa6, 0xe1, 0x83, 0xa3, 0xe1,
    0x83, 0x9b, 0xe1, 0x83, 0xa4, 0xe1, 0x83, 0x94, 0xe1, 0x83, 0x91, 0xe1,
    0x83, 0x98, 0x20, 0xe1, 0x83, 0x93, 0xe1, 0x83, 0x90, 0xe1, 0x83, 0x90,
    0xe1, 0x83, 0xa1, 0xe1, 0x83, 0x90, 0xe1, 0x83, 0xad, 0xe1, 0x83, 0xa7,
    0xe1, 0x83, 0x90, 0xe1, 0x83, 0x9c, 0xe1, 0x83, 0x90, 0x0a
};

// zlib-wrapped deflate (RFC 1950) of cInput at level 9
constexpr unsigned char cWrappedLevel9[] = {
    0x78, 0xda, 0x4d, 0x8b, 0xc9, 0x09, 0x80, 0x40, 0x10, 0x04, 0xff, 0x46,
    0x61, 0xae, 0x9d, 0xc0, 0x8a, 0x3e, 0x84, 0xdd, 0x87, 0x17, 0x8a, 0xe2,
    0x9a, 0x53, 0x65, 0xe2, 0xd0, 0x7e, 0x84, 0xa1, 0x99, 0xea, 0x03, 0x65,
    0x34, 0xa0, 0x0b, 0x3d, 0x28, 0xa1, 0xc5, 0x98, 0x5b, 0x74, 0x9a, 0x6f,
    0x74, 0xf8, 0x29, 0x68, 0x45, 0x13, 0x1a, 0x23, 0xdb, 0x0d, 0x31, 0xdd,
    0x5c, 0xef, 0x3e, 0xb7, 0x77, 0x31, 0x6e, 0xb6, 0xd6, 0xdf, 0x34, 0x35,
    0x2f, 0x7b, 0xa4, 0x44, 0x67
};

// zlib-wrapped deflate (RFC 1950) of cInput at level 1
constexpr unsigned char cWrappedLevel1[] = {
    0x78, 0x01, 0x4d, 0x8c, 0xbb, 0x0d, 0x80, 0x30, 0x14, 0x03, 0x7b, 0xa6,
    0x60, 0x57, 0x2f, 0x10, 0x04, 0x05, 0x52, 0x52, 0xf0, 0x13, 0x08, 0x44,
    0xd8, 0xe9, 0x36, 0xe1, 0xc9, 0x34, 0x74, 0x3e, 0xff, 0x50, 0x46, 0x03,
    0xba, 0xd0, 0x83, 0x12, 0x5a, 0x8c, 0xb9, 0x45, 0xa7, 0xf9, 0x46, 0x87,
    0x45, 0x41, 0x2b, 0x9a, 0xd0, 0x18, 0xd9, 0x6e, 0x88, 0xe9, 0xe6, 0x7a,
    0xf7, 0xb9, 0xbd, 0x8b, 0x71, 0x32, 0x5b, 0xd4, 0xdf, 0x34, 0x35, 0x2f,
    0x7b, 0xa4, 0x44, 0x67
};

// raw deflate (RFC 1951) of cInput at level 9 — cWrappedLevel9 without the 2-byte header and 4-byte Adler-32 trailer
constexpr unsigned char cRawLevel9[] = {
    0x4d, 0x8b, 0xc9, 0x09, 0x80, 0x40, 0x10, 0x04, 0xff, 0x46,
    0x61, 0xae, 0x9d, 0xc0, 0x8a, 0x3e, 0x84, 0xdd, 0x87, 0x17, 0x8a, 0xe2,
    0x9a, 0x53, 0x65, 0xe2, 0xd0, 0x7e, 0x84, 0xa1, 0x99, 0xea, 0x03, 0x65,
    0x34, 0xa0, 0x0b, 0x3d, 0x28, 0xa1, 0xc5, 0x98, 0x5b, 0x74, 0x9a, 0x6f,
    0x74, 0xf8, 0x29, 0x68, 0x45, 0x13, 0x1a, 0x23, 0xdb, 0x0d, 0x31, 0xdd,
    0x5c, 0xef, 0x3e, 0xb7, 0x77, 0x31, 0x6e, 0xb6, 0xd6, 0xdf, 0x34, 0x35,
    0x2f
};

// raw deflate (RFC 1951) of cInput at level 1 — cWrappedLevel1 without the 2-byte header and 4-byte Adler-32 trailer
constexpr unsigned char cRawLevel1[] = {
    0x4d, 0x8c, 0xbb, 0x0d, 0x80, 0x30, 0x14, 0x03, 0x7b, 0xa6,
    0x60, 0x57, 0x2f, 0x10, 0x04, 0x05, 0x52, 0x52, 0xf0, 0x13, 0x08, 0x44,
    0xd8, 0xe9, 0x36, 0xe1, 0xc9, 0x34, 0x74, 0x3e, 0xff, 0x50, 0x46, 0x03,
    0xba, 0xd0, 0x83, 0x12, 0x5a, 0x8c, 0xb9, 0x45, 0xa7, 0xf9, 0x46, 0x87,
    0x45, 0x41, 0x2b, 0x9a, 0xd0, 0x18, 0xd9, 0x6e, 0x88, 0xe9, 0xe6, 0x7a,
    0xf7, 0xb9, 0xbd, 0x8b, 0x71, 0x32, 0x5b, 0xd4, 0xdf, 0x34, 0x35, 0x2f
};

} // namespace

using ZlibCompressParameters = std::tuple<const unsigned char*, size_t, const unsigned char*, size_t, int, bool>;
class ZlibCompressTestFixture : public testing::TestWithParam<ZlibCompressParameters> {};

TEST_P( ZlibCompressTestFixture, ZlibCompress )
{
    const auto& [input, inputSize, output, outputSize, level, rawDeflate] = GetParam();

    const std::string inputStr( reinterpret_cast<const char*>( input ), inputSize );
    std::istringstream in( inputStr );

    const std::string outputStr( reinterpret_cast<const char*>( output ), outputSize );
    std::ostringstream out( outputStr );

    auto res = MR::zlibCompressStream( in, out, MR::ZlibCompressParams{ { .rawDeflate = rawDeflate }, level } );
    EXPECT_TRUE( res.has_value() );
    // FIXME: Python and MeshLib output data mismatch; note that MeshLib output is still valid
    //EXPECT_STREQ( out.str().c_str(), outputStr.c_str() );

    std::istringstream in2( out.str() );
    std::ostringstream out2;
    res = MR::zlibDecompressStream( in2, out2, MR::ZlibParams{ .rawDeflate = rawDeflate } );
    EXPECT_TRUE( res.has_value() );
    EXPECT_STREQ( out2.str().c_str(), inputStr.c_str() );
}

INSTANTIATE_TEST_SUITE_P( MRMesh, ZlibCompressTestFixture, testing::Values(
    ZlibCompressParameters { cInput, sizeof( cInput ), cWrappedLevel1, sizeof( cWrappedLevel1 ), 1, false },
    ZlibCompressParameters { cInput, sizeof( cInput ), cWrappedLevel9, sizeof( cWrappedLevel9 ), 9, false },
    ZlibCompressParameters { cInput, sizeof( cInput ), cRawLevel1,     sizeof( cRawLevel1 ),     1, true  },
    ZlibCompressParameters { cInput, sizeof( cInput ), cRawLevel9,     sizeof( cRawLevel9 ),     9, true  }
) );

using ZlibDecompressParameters = std::tuple<const unsigned char*, size_t, const unsigned char*, size_t, bool>;
class ZlibDecompressTestFixture : public testing::TestWithParam<ZlibDecompressParameters> {};

TEST_P( ZlibDecompressTestFixture, ZlibDecompress )
{
    const auto& [input, inputSize, output, outputSize, rawDeflate] = GetParam();

    const std::string inputStr( reinterpret_cast<const char*>( input ), inputSize );
    std::istringstream in( inputStr );

    const std::string outputStr( reinterpret_cast<const char*>( output ), outputSize );
    std::ostringstream out( outputStr );

    auto res = MR::zlibDecompressStream( in, out, MR::ZlibParams{ .rawDeflate = rawDeflate } );
    EXPECT_TRUE( res.has_value() );
    EXPECT_STREQ( out.str().c_str(), outputStr.c_str() );
}

INSTANTIATE_TEST_SUITE_P( MRMesh, ZlibDecompressTestFixture, testing::Values(
    ZlibDecompressParameters { cWrappedLevel1, sizeof( cWrappedLevel1 ), cInput, sizeof( cInput ), false },
    ZlibDecompressParameters { cWrappedLevel9, sizeof( cWrappedLevel9 ), cInput, sizeof( cInput ), false },
    ZlibDecompressParameters { cRawLevel1,     sizeof( cRawLevel1 ),     cInput, sizeof( cInput ), true  },
    ZlibDecompressParameters { cRawLevel9,     sizeof( cRawLevel9 ),     cInput, sizeof( cInput ), true  }
) );

TEST( MRMesh, ZlibCompressStats )
{
    const std::string inputStr( reinterpret_cast<const char*>( cInput ), sizeof( cInput ) );
    const uint32_t expectedCrc = crc32Ref( cInput, sizeof( cInput ) );

    struct Case { bool rawDeflate; int level; };
    const Case cases[] = {
        { true,  1 },
        { true,  9 },
        { false, 1 },
        { false, 9 },
    };

    for ( const auto& c : cases )
    {
        MR::ZlibCompressStats stats;
        std::istringstream in( inputStr );
        std::ostringstream out;
        auto res = MR::zlibCompressStream( in, out,
            MR::ZlibCompressParams{ { .rawDeflate = c.rawDeflate }, c.level, &stats } );
        EXPECT_TRUE( res.has_value() );

        // Engine-agnostic assertions: CRC and uncompressed size are defined by
        // the input; stats.compressedSize must equal out.str().size() (API
        // consistency); compressed output must be non-empty and smaller than
        // the input. We deliberately do NOT pin stats.compressedSize to the
        // size of cRawLevel*/cWrappedLevel* -- those reference blobs were
        // captured from stock zlib, and the exact compressed byte count drifts
        // from one deflate implementation to another (stock zlib vs zlib-ng
        // compat vs zlib-ng native vs libdeflate, all lossless).
        EXPECT_EQ( stats.crc32, expectedCrc );
        EXPECT_EQ( stats.uncompressedSize, sizeof( cInput ) );
        EXPECT_EQ( stats.compressedSize, out.str().size() );
        EXPECT_GT( stats.compressedSize, 0u );
        EXPECT_LT( stats.compressedSize, sizeof( cInput ) );
    }
}

TEST( MRMesh, ZlibCompressStatsEmpty )
{
    for ( bool rawDeflate : { false, true } )
    {
        MR::ZlibCompressStats stats;
        std::istringstream in;            // empty input
        std::ostringstream out;
        auto res = MR::zlibCompressStream( in, out,
            MR::ZlibCompressParams{ { .rawDeflate = rawDeflate }, /*level*/ -1, &stats } );
        EXPECT_TRUE( res.has_value() );

        EXPECT_EQ( stats.crc32, 0u );
        EXPECT_EQ( stats.uncompressedSize, 0u );
        // deflate emits at least a terminator block for Z_FINISH on empty input;
        // pin whatever zlib actually wrote.
        EXPECT_EQ( stats.compressedSize, out.str().size() );
        EXPECT_GT( stats.compressedSize, 0u );
    }
}
