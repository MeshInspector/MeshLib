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
Raster makeRaster( const Vector2i& size, ScalarType type, int channels, const std::vector<T>& values )
{
    Raster res{
        .info = {
            .dims = Vector3i( size.x, size.y, 1 ),
            .type = type,
            .channels = channels,
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
    info.channels = 3;
    EXPECT_EQ( info.dataSize(), 3u * 2 * 4 * 2 * 3 );
    info.channels = 0;
    EXPECT_EQ( info.dataSize(), 0u );
    info = { .dims = { 3, 2, 4 }, .type = ScalarType::Unknown };
    EXPECT_EQ( info.dataSize(), 0u );
    info = { .dims = { 3, 0, 1 }, .type = ScalarType::UInt8 };
    EXPECT_EQ( info.dataSize(), 0u );

    // the size that does not fit in size_t cannot match any data
    info = { .dims = { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF }, .type = ScalarType::Float64 };
    EXPECT_EQ( info.dataSize(), std::numeric_limits<size_t>::max() );
}

TEST( MRMesh, RasterToImageGray )
{
    // the values are scaled to [0, 255]; the rows of a raster go from top to bottom, the ones of an image from bottom to top
    const auto raster = makeRaster<uint8_t>( { 2, 2 }, ScalarType::UInt8, 1, { 0, 1, 2, 3 } );
    auto image = convertRasterToImage( raster );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->resolution, Vector2i( 2, 2 ) );
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 170, 170, 170 ), Color( 255, 255, 255 ),
        Color( 0, 0, 0 ), Color( 85, 85, 85 ),
    } ) );
}

TEST( MRMesh, RasterToImageScaledGray )
{
    // the values of other types are scaled from the range of non-NaN values
    const auto raster = makeRaster<float>( { 4, 1 }, ScalarType::Float32, 1, { -1.f, 1.f, 3.f, std::nanf( "" ) } );
    auto image = convertRasterToImage( raster );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( {
        Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ), Color::black()
    } ) );

    // infinite values are excluded from the range too, even the largest finite values do not overflow it
    const auto infinite = makeRaster<float>( { 4, 1 }, ScalarType::Float32, 1,
        { -std::numeric_limits<float>::infinity(), 1.f, 3.f, std::numeric_limits<float>::infinity() } );
    image = convertRasterToImage( infinite );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color::black(), Color( 0, 0, 0 ), Color( 255, 255, 255 ), Color::black() } ) );
    const auto extreme = makeRaster<double>( { 3, 1 }, ScalarType::Float64, 1,
        { std::numeric_limits<double>::lowest(), 0., std::numeric_limits<double>::max() } );
    image = convertRasterToImage( extreme );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ) } ) );

    const auto raster16 = makeRaster<int16_t>( { 3, 1 }, ScalarType::Int16, 1, { -300, 0, 300 } );
    image = convertRasterToImage( raster16 );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 127, 127, 127 ), Color( 255, 255, 255 ) } ) );
}

TEST( MRMesh, RasterToImageColor )
{
    const auto rgba = makeRaster<uint8_t>( { 2, 1 }, ScalarType::UInt8, 4, { 1, 2, 3, 4, 5, 6, 7, 8 } );
    auto image = convertRasterToImage( rgba );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 2, 3, 4 ), Color( 5, 6, 7, 8 ) } ) );

    const auto rgb = makeRaster<uint8_t>( { 2, 1 }, ScalarType::UInt8, 3, { 1, 2, 3, 4, 5, 6 } );
    image = convertRasterToImage( rgb );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 2, 3 ), Color( 4, 5, 6 ) } ) );
}

// wider integer components keep their high byte, taken as unsigned; floating-point components are clamped to [0, 1], NaN becomes 0
TEST( MRMesh, RasterToImageColorConversions )
{
    auto check = [] ( const Raster& raster, const Color& expected )
    {
        auto image = convertRasterToImage( raster );
        ASSERT_TRUE( image.has_value() ) << image.error();
        EXPECT_EQ( image->pixels, std::vector<Color>( { expected } ) );
    };
    check( makeRaster<uint16_t>( { 1, 1 }, ScalarType::UInt16, 3, { 0xFFFF, 0x80FF, 0x00FF } ), Color( 255, 128, 0 ) );
    check( makeRaster<int16_t>( { 1, 1 }, ScalarType::Int16, 3, { 20000, -1, 0 } ), Color( 78, 255, 0 ) );
    check( makeRaster<uint32_t>( { 1, 1 }, ScalarType::UInt32, 3, { 0xFF000000, 0x80FFFFFF, 0x00FFFFFF } ), Color( 255, 128, 0 ) );
    // the fourth channel is alpha, the channels after it are ignored
    check( makeRaster<float>( { 1, 1 }, ScalarType::Float32, 5, { 2.f, 0.5f, std::nanf( "" ), 1.f, 7.f } ), Color( 255, 127, 0, 255 ) );
}

TEST( MRMesh, RasterToImageErrors )
{
    auto raster = makeRaster<uint8_t>( { 2, 2 }, ScalarType::UInt8, 1, { 0, 1, 2 } );
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );

    raster.data.push_back( 3 );
    EXPECT_TRUE( convertRasterToImage( raster ).has_value() );

    auto twoLayers = makeRaster<uint8_t>( { 2, 1 }, ScalarType::UInt8, 1, { 0, 1, 2, 3 } );
    twoLayers.info.dims.z = 2;
    EXPECT_FALSE( convertRasterToImage( twoLayers ).has_value() );

    auto twoChannels = makeRaster<uint8_t>( { 2, 1 }, ScalarType::UInt8, 2, { 0, 1, 2, 3 } );
    EXPECT_FALSE( convertRasterToImage( twoChannels ).has_value() );

    raster.info.type = ScalarType::Unknown;
    EXPECT_FALSE( convertRasterToImage( raster ).has_value() );
}

} //namespace MR
