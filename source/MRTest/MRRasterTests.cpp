#include "MRMesh/MRRaster.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace MR
{

namespace
{

template <typename T>
Raster makeRaster( const Vector2i& size, ScalarType type, const std::vector<T>& values )
{
    Raster res{
        .info = {
            .dims = Vector3i( size.x, size.y, 1 ),
            .type = type,
        },
    };
    res.data.resize( values.size() * sizeof( T ) );
    std::memcpy( res.data.data(), values.data(), res.data.size() );
    return res;
}

} // namespace

TEST( MRMesh, RasterInfoDataSize )
{
    RasterInfo info{
        .dims = { 3, 2, 4 },
        .type = ScalarType::UInt16,
    };
    EXPECT_EQ( info.dataSize(), 3u * 2 * 4 * 2 );
    info.type = ScalarType::RGB8;
    EXPECT_EQ( info.dataSize(), 3u * 2 * 4 * 3 );
    info.type = ScalarType::Unknown;
    EXPECT_EQ( info.dataSize(), 0u );
    info = { .dims = { 3, 0, 1 }, .type = ScalarType::UInt8 };
    EXPECT_EQ( info.dataSize(), 0u );

    // the size that does not fit in size_t cannot match any data
    info = { .dims = { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF }, .type = ScalarType::Float64 };
    EXPECT_EQ( info.dataSize(), std::numeric_limits<size_t>::max() );
}

TEST( MRMesh, RasterToImageGray )
{
    // the rows of a raster go from top to bottom, the ones of an image from bottom to top
    const auto raster8 = makeRaster<uint8_t>( { 2, 2 }, ScalarType::UInt8, { 0, 50, 100, 255 } );
    auto image = convertRasterToImage( raster8 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->resolution, Vector2i( 2, 2 ) );
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 100, 100, 100 ), Color( 255, 255, 255 ),
        Color( 0, 0, 0 ), Color( 50, 50, 50 ),
    } ) );

    // 16-bit values keep the high byte, as libtiff's RGBA reader does
    const auto raster16 = makeRaster<uint16_t>( { 3, 1 }, ScalarType::UInt16, { 0x00FF, 0x8000, 0xFFFF } );
    image = convertRasterToImage( raster16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 128, 128, 128 ), Color( 255, 255, 255 ) } ) );
}

TEST( MRMesh, RasterToImageScaledGray )
{
    // the values of other types are scaled from the range of non-NaN values
    const auto raster = makeRaster<float>( { 4, 1 }, ScalarType::Float32, { -1.f, 1.f, 3.f, std::nanf( "" ) } );
    auto image = convertRasterToImage( raster );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ), Color::black()
    } ) );

    // infinite values are excluded from the range too, even the largest finite values do not overflow it
    const auto infinite = makeRaster<float>( { 4, 1 }, ScalarType::Float32,
        { -std::numeric_limits<float>::infinity(), 1.f, 3.f, std::numeric_limits<float>::infinity() } );
    image = convertRasterToImage( infinite );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color::black(), Color( 0, 0, 0 ), Color( 255, 255, 255 ), Color::black() } ) );
    const auto extreme = makeRaster<double>( { 3, 1 }, ScalarType::Float64,
        { std::numeric_limits<double>::lowest(), 0., std::numeric_limits<double>::max() } );
    image = convertRasterToImage( extreme );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ) } ) );

    const auto raster16 = makeRaster<int16_t>( { 3, 1 }, ScalarType::Int16, { -300, 0, 300 } );
    image = convertRasterToImage( raster16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ) } ) );
}

TEST( MRMesh, RasterToImageColor )
{
    const auto rgba = makeRaster<uint8_t>( { 2, 1 }, ScalarType::RGBA8, { 1, 2, 3, 4, 5, 6, 7, 8 } );
    auto image = convertRasterToImage( rgba );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 2, 3, 4 ), Color( 5, 6, 7, 8 ) } ) );

    const auto rgb = makeRaster<uint8_t>( { 2, 1 }, ScalarType::RGB8, { 1, 2, 3, 4, 5, 6 } );
    image = convertRasterToImage( rgb );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 2, 3 ), Color( 4, 5, 6 ) } ) );
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
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    EXPECT_EQ( raster->info, ( RasterInfo{ .dims = { 3, 2, 1 }, .type = ScalarType::RGBA8 } ) );
    ASSERT_EQ( raster->data.size(), 24u );
    // the first row of the raster is the top one, i.e. the last row of the image
    EXPECT_EQ( raster->data[0], 13 );
    EXPECT_EQ( raster->data[12], 1 );

    auto back = convertRasterToImage( *raster );
    ASSERT_TRUE( back.has_value() ) << back.error();
    EXPECT_EQ( back->resolution, image.resolution );
    EXPECT_EQ( back->pixels, image.pixels );
}

TEST( MRMesh, RasterToImageErrors )
{
    auto raster = makeRaster<uint8_t>( { 2, 2 }, ScalarType::UInt8, { 0, 1, 2 } );
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );

    raster.data.push_back( 3 );
    EXPECT_TRUE( convertRasterToImage( raster ).has_value() );

    auto twoLayers = makeRaster<uint8_t>( { 2, 1 }, ScalarType::UInt8, { 0, 1, 2, 3 } );
    twoLayers.info.dims.z = 2;
    EXPECT_FALSE( convertRasterToImage( twoLayers ).has_value() );

    raster.info.type = ScalarType::Unknown;
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );
}

TEST( MRMesh, ImageToRasterErrors )
{
    Image image{
        .pixels = { Color::red(), Color::green(), Color::blue() },
        .resolution = { 2, 2 },
    };
    EXPECT_FALSE( convertImageToRaster( image ).has_value() );
    image.resolution = { 4, -1 };
    EXPECT_FALSE( convertImageToRaster( image ).has_value() );
    image.resolution = { 3, 1 };
    EXPECT_TRUE( convertImageToRaster( image ).has_value() );
}

} //namespace MR
