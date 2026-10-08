#include "MRMesh/MRColor.h"
#include "MRMesh/MRScalarConvert.h"
#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsLoad.h"
#endif

#include <gtest/gtest.h>

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
#endif

} //namespace MR
