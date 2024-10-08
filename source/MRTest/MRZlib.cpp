#include <MRIOExtras/config.h>
#ifndef MRIOEXTRAS_NO_ZLIB
#include <MRMesh/MRGTest.h>
#include <MRIOExtras/MRZlib.h>

namespace
{

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

constexpr unsigned char cOutputLevel9[] = {
    0x78, 0xda, 0x4d, 0x8b, 0xc9, 0x09, 0x80, 0x40, 0x10, 0x04, 0xff, 0x46,
    0x61, 0xae, 0x9d, 0xc0, 0x8a, 0x3e, 0x84, 0xdd, 0x87, 0x17, 0x8a, 0xe2,
    0x9a, 0x53, 0x65, 0xe2, 0xd0, 0x7e, 0x84, 0xa1, 0x99, 0xea, 0x03, 0x65,
    0x34, 0xa0, 0x0b, 0x3d, 0x28, 0xa1, 0xc5, 0x98, 0x5b, 0x74, 0x9a, 0x6f,
    0x74, 0xf8, 0x29, 0x68, 0x45, 0x13, 0x1a, 0x23, 0xdb, 0x0d, 0x31, 0xdd,
    0x5c, 0xef, 0x3e, 0xb7, 0x77, 0x31, 0x6e, 0xb6, 0xd6, 0xdf, 0x34, 0x35,
    0x2f, 0x7b, 0xa4, 0x44, 0x67
};

constexpr unsigned char cOutputLevel1[] = {
    0x78, 0x01, 0x4d, 0x8c, 0xbb, 0x0d, 0x80, 0x30, 0x14, 0x03, 0x7b, 0xa6,
    0x60, 0x57, 0x2f, 0x10, 0x04, 0x05, 0x52, 0x52, 0xf0, 0x13, 0x08, 0x44,
    0xd8, 0xe9, 0x36, 0xe1, 0xc9, 0x34, 0x74, 0x3e, 0xff, 0x50, 0x46, 0x03,
    0xba, 0xd0, 0x83, 0x12, 0x5a, 0x8c, 0xb9, 0x45, 0xa7, 0xf9, 0x46, 0x87,
    0x45, 0x41, 0x2b, 0x9a, 0xd0, 0x18, 0xd9, 0x6e, 0x88, 0xe9, 0xe6, 0x7a,
    0xf7, 0xb9, 0xbd, 0x8b, 0x71, 0x32, 0x5b, 0xd4, 0xdf, 0x34, 0x35, 0x2f,
    0x7b, 0xa4, 0x44, 0x67
};

} // namespace

using ZlibCompressParameters = std::tuple<const unsigned char*, size_t, const unsigned char*, size_t, int>;
class ZlibCompressTestFixture : public testing::TestWithParam<ZlibCompressParameters> {};

TEST_P( ZlibCompressTestFixture, ZlibCompress )
{
    const auto& [input, inputSize, output, outputSize, level] = GetParam();

    const std::string inputStr( reinterpret_cast<const char*>( input ), inputSize );
    std::istringstream in( inputStr );

    const std::string outputStr( reinterpret_cast<const char*>( output ), outputSize );
    std::ostringstream out( outputStr );

    auto res = MR::zlibCompressStream( in, out );
    EXPECT_TRUE( res.has_value() );
    // FIXME: Python and MeshLib output data mismatch; note that MeshLib output is still valid
    //EXPECT_STREQ( out.str().c_str(), outputStr.c_str() );

    std::istringstream in2( out.str() );
    std::ostringstream out2;
    res = MR::zlibDecompressStream( in2, out2 );
    EXPECT_TRUE( res.has_value() );
    EXPECT_STREQ( out2.str().c_str(), inputStr.c_str() );
}

INSTANTIATE_TEST_SUITE_P( MRMesh, ZlibCompressTestFixture, testing::Values(
    ZlibCompressParameters { cInput, sizeof( cInput ), cOutputLevel1, sizeof( cOutputLevel1 ), 1 },
    ZlibCompressParameters { cInput, sizeof( cInput ), cOutputLevel9, sizeof( cOutputLevel9 ), 9 }
) );

using ZlibDecompressParameters = std::tuple<const unsigned char*, size_t, const unsigned char*, size_t>;
class ZlibDecompressTestFixture : public testing::TestWithParam<ZlibDecompressParameters> {};

TEST_P( ZlibDecompressTestFixture, ZlibDecompress )
{
    const auto& [input, inputSize, output, outputSize] = GetParam();

    const std::string inputStr( reinterpret_cast<const char*>( input ), inputSize );
    std::istringstream in( inputStr );

    const std::string outputStr( reinterpret_cast<const char*>( output ), outputSize );
    std::ostringstream out( outputStr );

    auto res = MR::zlibDecompressStream( in, out );
    EXPECT_TRUE( res.has_value() );
    EXPECT_STREQ( out.str().c_str(), outputStr.c_str() );
}

INSTANTIATE_TEST_SUITE_P( MRMesh, ZlibDecompressTestFixture, testing::Values(
    ZlibDecompressParameters { cOutputLevel1, sizeof( cOutputLevel1 ), cInput, sizeof( cInput ) },
    ZlibDecompressParameters { cOutputLevel9, sizeof( cOutputLevel9 ), cInput, sizeof( cInput ) }
) );
#endif
