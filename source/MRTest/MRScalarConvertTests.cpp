#include "MRMesh/MRColor.h"
#include "MRMesh/MRScalarConvert.h"
#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsLoad.h"
#endif

#include <gtest/gtest.h>

#include <limits>
#include <sstream>

namespace MR
{

TEST( MRMesh, ScalarTypeSize )
{
    EXPECT_EQ( getScalarTypeSize( ScalarType::UInt8 ), 1u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::Int16 ), 2u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::UInt32 ), 4u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::Float64 ), 8u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::Float32_4 ), 16u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::RGB8 ), 3u );
    EXPECT_EQ( getScalarTypeSize( ScalarType::RGBA8 ), sizeof( Color ) );
    EXPECT_EQ( getScalarTypeSize( ScalarType::Unknown ), 0u );
}

TEST( MRMesh, ScalarTypeMinMax )
{
    using Limits = std::pair<std::int64_t, std::uint64_t>;
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::UInt8 ), Limits( 0, 255u ) );
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::Int16 ), Limits( -32768, 32767u ) );
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::UInt64 ), Limits( 0, std::numeric_limits<std::uint64_t>::max() ) );
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::Int64 ),
        Limits( std::numeric_limits<std::int64_t>::lowest(), std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
    // the range of a color component
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::RGBA8 ), Limits( 0, 255u ) );
    EXPECT_EQ( getScalarTypeMinMax( ScalarType::Float64 ), Limits( 0, 0u ) );

    // getTypeConverter maps this range to [0, 1]
    const auto [min, max] = getScalarTypeMinMax( ScalarType::Int16 );
    const auto converter = getTypeConverter( ScalarType::Int16, max - min, min );
    const int16_t values[] = { -32768, 32767 };
    EXPECT_EQ( converter( ( const char* )&values[0] ), 0.f );
    EXPECT_EQ( converter( ( const char* )&values[1] ), 1.f );
    // and keeps floating-point values as they are
    const double value = -2.5;
    EXPECT_EQ( getTypeConverter( ScalarType::Float64, 0, 0 )( ( const char* )&value ), -2.5f );
}

TEST( MRMesh, ScalarTypeColorLuma )
{
    const uint8_t color[] = { 255, 128, 0, 7 };
    const auto* c = ( const char* )color;
    const float luma = 0.299f * 255 + 0.587f * 128;
    auto toFloat = [] ( auto v ) { return float( v ); };
    EXPECT_FLOAT_EQ( visitScalarType( toFloat, ScalarType::RGB8, c ), luma );
    // alpha does not change the value
    EXPECT_FLOAT_EQ( visitScalarType( toFloat, ScalarType::RGBA8, c ), luma );
    EXPECT_FLOAT_EQ( getTypeConverter( ScalarType::RGBA8, 255, 0 )( c ), luma / 255 );
}

#ifndef MESHLIB_NO_VOXELS
TEST( MRMesh, RawVoxelsColor )
{
    // red and blue voxels
    const uint8_t data[] = { 255, 0, 0, 0, 0, 255 };
    std::istringstream in( std::string( ( const char* )data, sizeof( data ) ) );
    auto vol = VoxelsLoad::fromRaw( in, {
        .dimensions = { 2, 1, 1 },
        .voxelSize = { 1, 1, 1 },
        .scalarType = ScalarType::RGB8,
    } );
    ASSERT_TRUE( vol.has_value() ) << vol.error();
    // the luma values are scaled to [0, 1] as for UInt8
    EXPECT_NEAR( vol->max, 0.299f, 1e-6f );
    EXPECT_NEAR( vol->min, 0.114f, 1e-6f );
}

TEST( MRMesh, RawVoxelsFloat )
{
    // Float64 and Float32_4 values are kept as they are, like Float32 ones
    const double float64[] = { 1.5, -2 };
    const float float32x4[] = { 0, 0, 0, 1.5f, 0, 0, 0, -2 };
    const std::pair<ScalarType, std::string> cases[] = {
        { ScalarType::Float64, std::string( ( const char* )float64, sizeof( float64 ) ) },
        { ScalarType::Float32_4, std::string( ( const char* )float32x4, sizeof( float32x4 ) ) },
    };
    for ( const auto& [scalarType, bytes] : cases )
    {
        std::istringstream in( bytes );
        auto vol = VoxelsLoad::fromRaw( in, {
            .dimensions = { 2, 1, 1 },
            .voxelSize = { 1, 1, 1 },
            .scalarType = scalarType,
        } );
        ASSERT_TRUE( vol.has_value() ) << vol.error();
        EXPECT_EQ( vol->min, -2.f );
        EXPECT_EQ( vol->max, 1.5f );
    }
}
#endif

} //namespace MR
