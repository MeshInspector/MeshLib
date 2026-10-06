#include "MRMesh/MRTiffIO.h"
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
#include "MRMesh/MRUniqueTemporaryFolder.h"

#include <gtest/gtest.h>

#include <bit>
#include <fstream>

namespace MR
{

// writes an uncompressed TIFF with 16-bit unsigned samples stored in tiles (more than one);
// the parts of the tiles outside the image are filled with the padding value
static void writeTiledTiff( const std::filesystem::path& path, const std::vector<uint16_t>& pixels,
    const Vector2i& size, const Vector2i& tileSize, uint16_t padding )
{
    constexpr uint32_t headerSize = 8;
    std::vector<uint16_t> tiles;
    std::vector<uint32_t> tileOffsets;
    for ( int ty = 0; ty < size.y; ty += tileSize.y )
    {
        for ( int tx = 0; tx < size.x; tx += tileSize.x )
        {
            tileOffsets.push_back( headerSize + uint32_t( tiles.size() * sizeof( uint16_t ) ) );
            for ( int y = ty; y < ty + tileSize.y; ++y )
                for ( int x = tx; x < tx + tileSize.x; ++x )
                    tiles.push_back( x < size.x && y < size.y ? pixels[x + y * size.x] : padding );
        }
    }
    const auto numTiles = uint32_t( tileOffsets.size() );
    const auto tileBytes = uint32_t( tileSize.x * tileSize.y * sizeof( uint16_t ) );
    const auto offsetsPos = headerSize + uint32_t( tiles.size() * sizeof( uint16_t ) );
    const auto byteCountsPos = offsetsPos + numTiles * uint32_t( sizeof( uint32_t ) );
    const auto ifdPos = byteCountsPos + numTiles * uint32_t( sizeof( uint32_t ) );

    std::ofstream out( path, std::ios::binary );
    auto write = [&] ( auto v ) { out.write( ( const char* )&v, sizeof( v ) ); };
    // all numbers are written in the native byte order
    out.write( std::endian::native == std::endian::little ? "II" : "MM", 2 );
    write( uint16_t( 42 ) );
    write( ifdPos );
    out.write( ( const char* )tiles.data(), tiles.size() * sizeof( uint16_t ) );
    for ( auto offset : tileOffsets )
        write( offset );
    for ( uint32_t i = 0; i < numTiles; ++i )
        write( tileBytes );

    // image file directory, the entries are sorted by tag; a single value is stored in the entry itself
    constexpr uint16_t SHORT = 3, LONG = 4;
    auto writeEntry = [&] ( uint16_t tag, uint16_t type, uint32_t count, uint32_t value )
    {
        write( tag );
        write( type );
        write( count );
        if ( type == SHORT )
        {
            write( uint16_t( value ) );
            write( uint16_t( 0 ) );
        }
        else
        {
            write( value );
        }
    };
    write( uint16_t( 10 ) );
    writeEntry( 256, LONG, 1, uint32_t( size.x ) ); // ImageWidth
    writeEntry( 257, LONG, 1, uint32_t( size.y ) ); // ImageLength
    writeEntry( 258, SHORT, 1, 16 ); // BitsPerSample
    writeEntry( 259, SHORT, 1, 1 ); // Compression: none
    writeEntry( 262, SHORT, 1, 1 ); // PhotometricInterpretation: BlackIsZero
    writeEntry( 277, SHORT, 1, 1 ); // SamplesPerPixel
    writeEntry( 322, LONG, 1, uint32_t( tileSize.x ) ); // TileWidth
    writeEntry( 323, LONG, 1, uint32_t( tileSize.y ) ); // TileLength
    writeEntry( 324, LONG, numTiles, offsetsPos ); // TileOffsets
    writeEntry( 325, LONG, numTiles, byteCountsPos ); // TileByteCounts
    write( uint32_t( 0 ) ); // no next directory
}

TEST( MRMesh, TiffTiledPartialTiles )
{
    // the tiles in the last column and in the last row extend beyond the image
    const Vector2i size{ 20, 18 };
    const Vector2i tileSize{ 16, 16 };
    std::vector<uint16_t> pixels( size_t( size.x ) * size.y );
    for ( int y = 0; y < size.y; ++y )
        for ( int x = 0; x < size.x; ++x )
            pixels[x + y * size.x] = uint16_t( 100 * y + x );

    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "tiled.tif";
    writeTiledTiff( path, pixels, size, tileSize, 0xFFFF );

    auto params = readTiffParameters( path );
    ASSERT_TRUE( params.has_value() ) << params.error();
    EXPECT_EQ( params->imageSize, size );
    EXPECT_TRUE( params->tiled );
    EXPECT_EQ( params->tileSize, tileSize );

    std::vector<uint16_t> raw( pixels.size(), 0xABCD );
    RawTiffOutput rawOutput{
        .bytes = ( uint8_t* )raw.data(),
        .size = raw.size() * sizeof( uint16_t ),
        .convertToFloat = false,
    };
    auto res = readRawTiff( path, rawOutput );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_EQ( raw, pixels );

    std::vector<float> floats( pixels.size() );
    RawTiffOutput floatOutput{
        .bytes = ( uint8_t* )floats.data(),
        .size = floats.size() * sizeof( float ),
    };
    res = readRawTiff( path, floatOutput );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_EQ( floats, std::vector<float>( pixels.begin(), pixels.end() ) );
}

} //namespace MR

#endif //!__EMSCRIPTEN__ && !MRMESH_NO_TIFF
