#include "MRMesh/MRTiffIO.h"
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
#include "MRMesh/MRImage.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRIOExtras/MRTiff.h"

#include <gtest/gtest.h>

namespace MR
{

// Cyrillic "test" and a Latin letter with diaeresis: no single ANSI code page has both
static const std::filesystem::path cNonAsciiName = u8"\u0442\u0435\u0441\u0442_\u00fc";

TEST( MRMesh, TiffNonAsciiPath )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto dir = tmpFolder / cNonAsciiName;
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::create_directory( dir, ec ) ) << systemToUtf8( ec.message() );
    const auto path = dir / "img.tif";

    const BaseTiffParameters baseParams{
        .sampleType = BaseTiffParameters::SampleType::Float,
        .valueType = BaseTiffParameters::ValueType::Scalar,
        .bytesPerSample = sizeof( float ),
        .imageSize = { 3, 2 },
    };
    const std::vector<float> values{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    auto saveRes = writeRawTiff( ( const uint8_t* )values.data(), path, { .baseParams = baseParams } );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    EXPECT_TRUE( std::filesystem::is_regular_file( path, ec ) );

    EXPECT_TRUE( isTIFFFile( path ) );

    auto params = readTiffParameters( path );
    ASSERT_TRUE( params.has_value() ) << params.error();
    EXPECT_EQ( BaseTiffParameters( *params ), baseParams );

    std::vector<float> loaded( values.size() );
    RawTiffOutput output{
        .bytes = ( uint8_t* )loaded.data(),
        .size = loaded.size() * sizeof( float ),
    };
    auto loadRes = readRawTiff( path, output );
    ASSERT_TRUE( loadRes.has_value() ) << loadRes.error();
    EXPECT_EQ( loaded, values );
}

#ifndef MRIOEXTRAS_NO_TIFF
TEST( MRMesh, TiffImageNonAsciiPath )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto dir = tmpFolder / cNonAsciiName;
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::create_directory( dir, ec ) ) << systemToUtf8( ec.message() );
    const auto path = dir / "img.tif";

    const Image image{
        .pixels = { Color::red(), Color::green(), Color::blue(), Color::white() },
        .resolution = { 2, 2 },
    };
    auto saveRes = ImageSave::toTiff( image, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    EXPECT_TRUE( std::filesystem::is_regular_file( path, ec ) );

    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->resolution, image.resolution );
}
#endif //!MRIOEXTRAS_NO_TIFF

} //namespace MR

#endif //!__EMSCRIPTEN__ && !MRMESH_NO_TIFF
