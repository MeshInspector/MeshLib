#include "MRMesh/MRRaster.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace MR
{

namespace
{

template <typename T>
Raster makeRaster( const Vector2i& resolution, int channels, ScalarType sampleType, const std::vector<T>& samples )
{
    Raster res{
        .info = {
            .resolution = resolution,
            .channels = channels,
            .sampleType = sampleType,
        },
    };
    res.data.resize( samples.size() * sizeof( T ) );
    std::memcpy( res.data.data(), samples.data(), res.data.size() );
    return res;
}

} // namespace

TEST( MRMesh, RasterInfoSizes )
{
    RasterInfo info{
        .resolution = { 3, 2 },
        .channels = 4,
        .sampleType = ScalarType::UInt16,
    };
    EXPECT_EQ( info.sampleSize(), 2u );
    EXPECT_EQ( info.dataSize(), 3u * 2 * 4 * 2 );

    info.sampleType = ScalarType::Float64;
    EXPECT_EQ( info.sampleSize(), 8u );
    info.sampleType = ScalarType::Float32_4;
    EXPECT_EQ( info.sampleSize(), 0u );
    EXPECT_EQ( info.dataSize(), 0u );
}

TEST( MRMesh, RasterToImageGray )
{
    // the rows of a raster go from top to bottom, the ones of an image from bottom to top
    const auto raster8 = makeRaster<uint8_t>( { 2, 2 }, 1, ScalarType::UInt8, { 0, 50, 100, 255 } );
    auto image = convertRasterToImage( raster8 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->resolution, Vector2i( 2, 2 ) );
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 100, 100, 100 ), Color( 255, 255, 255 ),
        Color( 0, 0, 0 ), Color( 50, 50, 50 ),
    } ) );

    // 16-bit gray values keep the high byte, as libtiff's RGBA reader does
    const auto raster16 = makeRaster<uint16_t>( { 3, 1 }, 1, ScalarType::UInt16, { 0x00FF, 0x8000, 0xFFFF } );
    image = convertRasterToImage( raster16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 128, 128, 128 ), Color( 255, 255, 255 ) } ) );

    auto minIsWhite = raster8;
    minIsWhite.info.minIsWhite = true;
    image = convertRasterToImage( minIsWhite );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 155, 155, 155 ), Color( 0, 0, 0 ),
        Color( 255, 255, 255 ), Color( 205, 205, 205 ),
    } ) );

    // gray and alpha
    const auto grayAlpha = makeRaster<uint8_t>( { 2, 1 }, 2, ScalarType::UInt8, { 10, 20, 30, 40 } );
    image = convertRasterToImage( grayAlpha );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 10, 10, 10, 20 ), Color( 30, 30, 30, 40 ) } ) );
}

TEST( MRMesh, RasterToImageScaledGray )
{
    // the values of other types are scaled from their valid range, invalid values become black
    auto raster = makeRaster<float>( { 5, 1 }, 1, ScalarType::Float32, { -1.f, 1.f, 3.f, -1000.f, std::nanf( "" ) } );
    raster.info.noData = -1000.;
    auto image = convertRasterToImage( raster );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ), Color::black(), Color::black()
    } ) );

    const auto raster16 = makeRaster<int16_t>( { 3, 1 }, 1, ScalarType::Int16, { -300, 0, 300 } );
    image = convertRasterToImage( raster16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ) } ) );
}

TEST( MRMesh, RasterToImageColor )
{
    // 16-bit color and alpha samples are rounded, as libtiff's RGBA reader does
    const auto rgba16 = makeRaster<uint16_t>( { 1, 1 }, 4, ScalarType::UInt16, { 0xFFFF, 0x8000, 0x0081, 0x0000 } );
    auto image = convertRasterToImage( rgba16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 128, 1, 0 ) } ) );

    // floating-point samples are clamped to [0, 1], the channels after the fourth one are ignored
    const auto rgbFloat = makeRaster<float>( { 1, 1 }, 5, ScalarType::Float32, { 2.f, 0.5f, -1.f, 1.f, 7.f } );
    image = convertRasterToImage( rgbFloat );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 127, 0, 255 ) } ) );

    const auto rgb8 = makeRaster<uint8_t>( { 1, 1 }, 3, ScalarType::UInt8, { 1, 2, 3 } );
    image = convertRasterToImage( rgb8 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 2, 3 ) } ) );
}

TEST( MRMesh, RasterToImagePalette )
{
    auto raster = makeRaster<uint8_t>( { 3, 1 }, 1, ScalarType::UInt8, { 1, 0, 7 } );
    raster.info.palette = { Color::red(), Color::green() };
    auto image = convertRasterToImage( raster );
    ASSERT_TRUE( image.has_value() ) << image.error();
    // the indices outside of the palette become black
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color::green(), Color::red(), Color::black() } ) );
}

TEST( MRMesh, RasterImageRoundTrip )
{
    const Image image{
        .pixels = {
            Color( 1, 2, 3, 4 ), Color( 5, 6, 7, 8 ), Color( 9, 10, 11, 12 ),
            Color( 13, 14, 15, 16 ), Color( 17, 18, 19, 20 ), Color( 21, 22, 23, 24 ),
        },
        .resolution = { 3, 2 },
    };
    const auto raster = convertImageToRaster( image );
    EXPECT_EQ( raster.info.resolution, image.resolution );
    EXPECT_EQ( raster.info.channels, 4 );
    EXPECT_EQ( raster.info.sampleType, ScalarType::UInt8 );
    ASSERT_EQ( raster.data.size(), 24u );
    // the first row of the raster is the top one, i.e. the last row of the image
    EXPECT_EQ( raster.data[0], 13 );
    EXPECT_EQ( raster.data[12], 1 );

    auto back = convertRasterToImage( raster );
    ASSERT_TRUE( back.has_value() ) << back.error();
    EXPECT_EQ( back->resolution, image.resolution );
    EXPECT_EQ( back->pixels, image.pixels );
}

TEST( MRMesh, RasterToImageErrors )
{
    auto raster = makeRaster<uint8_t>( { 2, 2 }, 1, ScalarType::UInt8, { 0, 1, 2 } );
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );

    raster.data.push_back( 3 );
    EXPECT_TRUE( convertRasterToImage( raster ).has_value() );

    raster.info.sampleType = ScalarType::Unknown;
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );
}

} //namespace MR
