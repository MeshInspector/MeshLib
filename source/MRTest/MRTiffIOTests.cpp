#include "MRMesh/MRTiffIO.h"
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRDistanceMapLoad.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRRasterLoad.h"
#include "MRMesh/MRRasterSave.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRIOExtras/MRTiff.h"
#include "MRPch/MRFmt.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>

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

// layout of a hand-written TIFF file
struct TestTiffLayout
{
    Vector2i size;
    uint16_t bitsPerSample = 8;
    uint16_t samplesPerPixel = 1;
    uint16_t sampleFormat = 1; // unsigned integer
    uint16_t photometric = 1; // BlackIsZero
    // the image is stored in tiles of this size if it is not zero, otherwise in strips
    Vector2i tileSize;
    // the number of rows in a strip, zero means a single strip
    int rowsPerStrip = 0;
    std::vector<uint16_t> extraSamples;
    // 3 * 2^bitsPerSample values for a palette image
    std::vector<uint16_t> colorMap;
    // the image file directory goes before the pixels, so that the file can be opened after its end is cut off
    bool directoryFirst = false;
};

// writes an uncompressed TIFF file in the native byte order; `samples` holds the pixels in the stored order:
// rows one after another, the samples of a pixel together, each sample takes bitsPerSample / 8 bytes;
// the parts of the tiles outside the image are filled with 0xFF bytes
static void writeTestTiff( const std::filesystem::path& path, const TestTiffLayout& layout, const std::vector<uint8_t>& samples )
{
    const size_t sampleSize = layout.bitsPerSample / 8;
    const size_t samplesPerPixel = layout.samplesPerPixel;
    const auto width = size_t( layout.size.x );
    const auto height = size_t( layout.size.y );
    ASSERT_EQ( samples.size(), width * height * samplesPerPixel * sampleSize );
    const bool tiled = layout.tileSize.x > 0;
    const auto chunkWidth = tiled ? size_t( layout.tileSize.x ) : width;
    const auto chunkHeight = tiled ? size_t( layout.tileSize.y ) : ( layout.rowsPerStrip > 0 ? size_t( layout.rowsPerStrip ) : height );

    // the pixels by chunks, their offsets are relative to the start of the pixels until the position of the pixels is known
    std::vector<uint8_t> pixels;
    std::vector<uint32_t> chunkOffsets, chunkSizes;
    const auto pixelSize = samplesPerPixel * sampleSize;
    for ( size_t y0 = 0; y0 < height; y0 += chunkHeight )
    {
        for ( size_t x0 = 0; x0 < width; x0 += chunkWidth )
        {
            chunkOffsets.push_back( uint32_t( pixels.size() ) );
            // the last strip ends with the last row, tiles are always complete
            const auto yEnd = tiled ? y0 + chunkHeight : std::min( y0 + chunkHeight, height );
            for ( size_t y = y0; y < yEnd; ++y )
            {
                for ( size_t x = x0; x < x0 + chunkWidth; ++x )
                {
                    if ( x < width && y < height )
                        pixels.insert( pixels.end(), samples.data() + ( y * width + x ) * pixelSize, samples.data() + ( y * width + x + 1 ) * pixelSize );
                    else
                        pixels.insert( pixels.end(), pixelSize, uint8_t( 0xFF ) );
                }
            }
            chunkSizes.push_back( uint32_t( pixels.size() - chunkOffsets.back() ) );
        }
    }

    constexpr uint16_t SHORT = 3, LONG = 4;
    struct Entry
    {
        uint16_t tag;
        uint16_t type;
        std::vector<uint32_t> values;
    };
    std::vector<Entry> entries{
        { 256, LONG, { uint32_t( width ) } }, // ImageWidth
        { 257, LONG, { uint32_t( height ) } }, // ImageLength
        { 258, SHORT, std::vector<uint32_t>( samplesPerPixel, layout.bitsPerSample ) }, // BitsPerSample
        { 259, SHORT, { 1 } }, // Compression: none
        { 262, SHORT, { layout.photometric } }, // PhotometricInterpretation
        { 277, SHORT, { layout.samplesPerPixel } }, // SamplesPerPixel
        { 339, SHORT, std::vector<uint32_t>( samplesPerPixel, layout.sampleFormat ) }, // SampleFormat
    };
    if ( tiled )
    {
        entries.push_back( { 322, LONG, { uint32_t( chunkWidth ) } } ); // TileWidth
        entries.push_back( { 323, LONG, { uint32_t( chunkHeight ) } } ); // TileLength
        entries.push_back( { 324, LONG, chunkOffsets } ); // TileOffsets
        entries.push_back( { 325, LONG, chunkSizes } ); // TileByteCounts
    }
    else
    {
        entries.push_back( { 273, LONG, chunkOffsets } ); // StripOffsets
        entries.push_back( { 278, LONG, { uint32_t( chunkHeight ) } } ); // RowsPerStrip
        entries.push_back( { 279, LONG, chunkSizes } ); // StripByteCounts
    }
    if ( !layout.colorMap.empty() )
        entries.push_back( { 320, SHORT, { layout.colorMap.begin(), layout.colorMap.end() } } ); // ColorMap
    if ( !layout.extraSamples.empty() )
        entries.push_back( { 338, SHORT, { layout.extraSamples.begin(), layout.extraSamples.end() } } ); // ExtraSamples
    std::sort( entries.begin(), entries.end(), [] ( const Entry& a, const Entry& b ) { return a.tag < b.tag; } );

    auto valueBytes = [&] ( const Entry& entry )
    {
        std::vector<uint8_t> bytes;
        for ( auto v : entry.values )
        {
            if ( entry.type == SHORT )
            {
                const auto s = uint16_t( v );
                bytes.insert( bytes.end(), ( const uint8_t* )&s, ( const uint8_t* )&s + sizeof( s ) );
            }
            else
            {
                bytes.insert( bytes.end(), ( const uint8_t* )&v, ( const uint8_t* )&v + sizeof( v ) );
            }
        }
        return bytes;
    };

    // the image file directory and the pixels start on word boundaries; the values longer than 4 bytes are stored after the directory
    auto even = [] ( size_t v ) { return v + v % 2; };
    size_t externalSize = 0;
    for ( const auto& entry : entries )
        if ( const auto size = valueBytes( entry ).size(); size > 4 )
            externalSize += even( size );
    const auto directorySize = 2 + 12 * entries.size() + 4 + externalSize;
    const auto directoryPos = layout.directoryFirst ? size_t( 8 ) : 8 + even( pixels.size() );
    const auto pixelsPos = layout.directoryFirst ? 8 + even( directorySize ) : size_t( 8 );
    for ( auto& entry : entries )
        if ( entry.tag == 273 || entry.tag == 324 ) // StripOffsets, TileOffsets
            for ( auto& offset : entry.values )
                offset += uint32_t( pixelsPos );

    std::vector<uint8_t> directory;
    auto append = [&] ( const void* data, size_t size )
    {
        directory.insert( directory.end(), ( const uint8_t* )data, ( const uint8_t* )data + size );
    };
    auto write16 = [&] ( uint16_t v ) { append( &v, sizeof( v ) ); };
    auto write32 = [&] ( uint32_t v ) { append( &v, sizeof( v ) ); };
    const auto externalPos = uint32_t( directoryPos + 2 + 12 * entries.size() + 4 );
    std::vector<uint8_t> external;
    write16( uint16_t( entries.size() ) );
    for ( const auto& entry : entries )
    {
        write16( entry.tag );
        write16( entry.type );
        write32( uint32_t( entry.values.size() ) );
        const auto bytes = valueBytes( entry );
        if ( bytes.size() <= 4 )
        {
            // the value is stored in the entry itself, padded with zeros to 4 bytes
            uint32_t value = 0;
            std::memcpy( &value, bytes.data(), bytes.size() );
            write32( value );
        }
        else
        {
            write32( externalPos + uint32_t( external.size() ) );
            external.insert( external.end(), bytes.begin(), bytes.end() );
            if ( external.size() % 2 != 0 )
                external.push_back( 0 );
        }
    }
    write32( 0 ); // no next directory
    directory.insert( directory.end(), external.begin(), external.end() );

    // all numbers are written in the native byte order
    std::vector<uint8_t> file( 8 );
    std::memcpy( file.data(), std::endian::native == std::endian::little ? "II" : "MM", 2 );
    const uint16_t magic = 42;
    std::memcpy( file.data() + 2, &magic, sizeof( magic ) );
    const auto directoryOffset = uint32_t( directoryPos );
    std::memcpy( file.data() + 4, &directoryOffset, sizeof( directoryOffset ) );
    const auto& first = layout.directoryFirst ? directory : pixels;
    const auto& second = layout.directoryFirst ? pixels : directory;
    file.insert( file.end(), first.begin(), first.end() );
    file.resize( layout.directoryFirst ? pixelsPos : directoryPos );
    file.insert( file.end(), second.begin(), second.end() );

    std::ofstream out( path, std::ios::binary );
    out.write( ( const char* )file.data(), std::streamsize( file.size() ) );
}

template <typename T>
static std::vector<uint8_t> toBytes( const std::vector<T>& values )
{
    std::vector<uint8_t> res( values.size() * sizeof( T ) );
    std::memcpy( res.data(), values.data(), res.size() );
    return res;
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
    // the tiles are padded with 0xFFFF
    writeTestTiff( path, { .size = size, .bitsPerSample = 16, .tileSize = tileSize }, toBytes( pixels ) );

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
    EXPECT_EQ( loaded->pixels, image.pixels );
}

// floating-point samples are scaled to [0, 255]
TEST( MRMesh, TiffImageFloat )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "float.tif";

    const std::vector<float> values{
        0.f, 1.f, // top row
        2.f, 3.f, // bottom row
    };
    auto saveRes = writeRawTiff( ( const uint8_t* )values.data(), path, { .baseParams = {
        .sampleType = BaseTiffParameters::SampleType::Float,
        .valueType = BaseTiffParameters::ValueType::Scalar,
        .bytesPerSample = sizeof( float ),
        .imageSize = { 2, 2 },
    } } );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->resolution, Vector2i( 2, 2 ) );
    // the values are scaled to [0, 255], and the rows of Image go from bottom to top;
    // writeRawTiff marks the file with PHOTOMETRIC_MINISWHITE, which is ignored for floating-point samples
    const std::vector<Color> expected{
        Color( 170, 170, 170 ), Color( 255, 255, 255 ),
        Color( 0, 0, 0 ), Color( 85, 85, 85 ),
    };
    EXPECT_EQ( loaded->pixels, expected );
}

// colors with alpha are loaded as saved
TEST( MRMesh, TiffImageAlpha )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "alpha.tif";

    const Image image{
        .pixels = { Color( 255, 0, 0, 128 ), Color( 1, 2, 3, 4 ), Color( 200, 100, 50, 0 ), Color( 7, 8, 9, 255 ) },
        .resolution = { 2, 2 },
    };
    auto saveRes = ImageSave::toTiff( image, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->pixels, image.pixels );
}

TEST( MRMesh, TiffRasterRoundTrip )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    for ( auto type : { ScalarType::UInt8, ScalarType::Int8, ScalarType::UInt16, ScalarType::Int16, ScalarType::UInt32, ScalarType::Int32,
        ScalarType::UInt64, ScalarType::Int64, ScalarType::Float32, ScalarType::Float64, ScalarType::RGB8, ScalarType::RGBA8 } )
    {
        Raster raster{
            .info = {
                .dims = { 5, 3, 1 },
                .type = type,
            },
        };
        raster.data.resize( raster.info.dataSize() );
        for ( size_t i = 0; i < raster.data.size(); ++i )
            raster.data[i] = uint8_t( i * 37 + 11 );

        const auto path = tmpFolder / fmt::format( "raster{}.tif", int( type ) );
        auto saveRes = RasterSave::toAnySupportedFormat( raster, path );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

        auto loaded = RasterLoad::fromAnySupportedFormat( path );
        ASSERT_TRUE( loaded.has_value() ) << loaded.error();
        EXPECT_EQ( loaded->info, raster.info );
        EXPECT_EQ( loaded->data, raster.data );

        auto info = RasterLoad::infoFromAnySupportedFormat( path );
        ASSERT_TRUE( info.has_value() ) << info.error();
        EXPECT_EQ( *info, raster.info );
    }
}

// a hand-written TIFF file with the values expected in the raster
struct TestTiffCase
{
    const char* name;
    TestTiffLayout layout;
    std::vector<uint8_t> samples;
    ScalarType type;
    std::vector<uint8_t> expected;
};

static void checkTestTiffCases( const std::vector<TestTiffCase>& cases )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    for ( const auto& c : cases )
    {
        const auto path = tmpFolder / ( std::string( c.name ) + ".tif" );
        writeTestTiff( path, c.layout, c.samples );
        auto loaded = RasterLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << c.name << ": " << loaded.error();
        EXPECT_EQ( loaded->info, ( RasterInfo{ .dims = Vector3i( c.layout.size.x, c.layout.size.y, 1 ), .type = c.type } ) ) << c.name;
        EXPECT_EQ( loaded->data, c.expected ) << c.name;
    }
}

// RGB samples wider than 8 bits keep their high byte, taken as unsigned; floating-point samples are clamped to [0, 1], NaN becomes 0
TEST( MRMesh, TiffRasterColorConversions )
{
    checkTestTiffCases( {
        { "rgb16", { .size = { 1, 1 }, .bitsPerSample = 16, .samplesPerPixel = 3, .photometric = 2 },
            toBytes( std::vector<uint16_t>{ 0xFFFF, 0x80FF, 0x00FF } ), ScalarType::RGB8, { 255, 128, 0 } },
        { "rgb16s", { .size = { 1, 1 }, .bitsPerSample = 16, .samplesPerPixel = 3, .sampleFormat = 2, .photometric = 2 },
            toBytes( std::vector<int16_t>{ 20000, -1, 0 } ), ScalarType::RGB8, { 78, 255, 0 } },
        { "rgb32", { .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 3, .photometric = 2 },
            toBytes( std::vector<uint32_t>{ 0xFF000000, 0x80FFFFFF, 0x00FFFFFF } ), ScalarType::RGB8, { 255, 128, 0 } },
        // the fourth sample is alpha, the samples after it are ignored
        { "rgbaf", { .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 5, .sampleFormat = 3, .photometric = 2, .extraSamples = { 2, 0 } },
            toBytes( std::vector<float>{ 2.f, 0.5f, std::nanf( "" ), 1.f, 7.f } ), ScalarType::RGBA8, { 255, 127, 0, 255 } },
    } );
}

// the formats without plain sample values are decoded by libtiff to RGBA8 values
TEST( MRMesh, TiffRasterDecoded )
{
    // red, green and blue colors for the indices 0, 1, 2, the rest is black
    std::vector<uint16_t> colorMap( 3 * 256, 0 );
    colorMap[0] = colorMap[256 + 1] = colorMap[512 + 2] = 0xFFFF;
    checkTestTiffCases( {
        // white, black and cyan
        { "cmyk", { .size = { 3, 1 }, .samplesPerPixel = 4, .photometric = 5 }, { 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0 },
            ScalarType::RGBA8, { 255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255 } },
        { "palette", { .size = { 3, 1 }, .photometric = 3, .colorMap = colorMap }, { 2, 0, 1 },
            ScalarType::RGBA8, { 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255 } },
        { "grayAlpha", { .size = { 2, 1 }, .samplesPerPixel = 2, .extraSamples = { 2 } }, { 10, 0, 20, 255 },
            ScalarType::RGBA8, { 10, 10, 10, 0, 20, 20, 20, 255 } },
    } );
}

// the samples are read from strips and tiles, the tiles of the last column and of the last row extend beyond the image
TEST( MRMesh, TiffRasterStoredLayouts )
{
    const Vector2i size{ 5, 3 };
    std::vector<uint8_t> rgb( size_t( size.x ) * size.y * 3 );
    for ( size_t i = 0; i < rgb.size(); ++i )
        rgb[i] = uint8_t( i * 7 + 3 );
    std::vector<uint16_t> gray( size_t( size.x ) * size.y );
    for ( size_t i = 0; i < gray.size(); ++i )
        gray[i] = uint16_t( 1000 * i + 7 );
    checkTestTiffCases( {
        { "strip", { .size = size, .samplesPerPixel = 3, .photometric = 2 }, rgb, ScalarType::RGB8, rgb },
        { "strips", { .size = size, .samplesPerPixel = 3, .photometric = 2, .rowsPerStrip = 2 }, rgb, ScalarType::RGB8, rgb },
        { "tiles", { .size = size, .samplesPerPixel = 3, .photometric = 2, .tileSize = { 4, 2 } }, rgb, ScalarType::RGB8, rgb },
        { "grayTiles", { .size = size, .bitsPerSample = 16, .tileSize = { 4, 2 } }, toBytes( gray ), ScalarType::UInt16, toBytes( gray ) },
    } );
}

// the formats that libtiff cannot decode are read as stored, as the old image loader read them
TEST( MRMesh, TiffRasterUndecodedFormat )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "linearraw.tif";

    writeTestTiff( path, { .size = { 2, 1 }, .bitsPerSample = 16, .photometric = 34892 }, toBytes( std::vector<uint16_t>{ 0x0100, 0xFF00 } ) ); // LinearRaw
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::UInt16 );
    EXPECT_EQ( loaded->data, toBytes( std::vector<uint16_t>{ 0x0100, 0xFF00 } ) );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 1, 1, 1 ), Color( 255, 255, 255 ) } ) );
}

// the raster keeps the stored gray values, libtiff inverts them in the image
TEST( MRMesh, TiffRasterMinIsWhite )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "white.tif";

    writeTestTiff( path, { .size = { 2, 1 }, .photometric = 0 }, { 0, 200 } );
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::UInt8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 0, 200 } ) );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 255, 255 ), Color( 55, 55, 55 ) } ) );
}

TEST( MRMesh, TiffRasterErrors )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // checks that loading fails with an error containing given text
    auto expectError = [] ( const auto& res, const std::string& text )
    {
        ASSERT_FALSE( res.has_value() ) << text;
        EXPECT_NE( res.error().find( text ), std::string::npos ) << res.error();
    };

    // libtiff's RGBA reader does not support 12-bit samples either
    const auto path12 = tmpFolder / "gray12.tif";
    writeTestTiff( path12, { .size = { 2, 1 }, .bitsPerSample = 12 }, { 1, 2 } );
    expectError( RasterLoad::fromTiff( path12 ), "12-bit" );
    expectError( RasterLoad::infoFromTiff( path12 ), "12-bit" );
    expectError( ImageLoad::fromTiff( path12 ), "12-bit" );

    const auto notTiff = tmpFolder / "text.tif";
    std::ofstream( notTiff ) << "not a TIFF file";
    expectError( RasterLoad::fromTiff( notTiff ), "Cannot read file" );

    // the file is cut off in the last strip or tile, the directory goes first, so the file still opens
    const auto cutOff = [] ( const std::filesystem::path& path )
    {
        std::error_code ec;
        const auto size = std::filesystem::file_size( path, ec );
        std::filesystem::resize_file( path, size - 2, ec );
    };
    const auto strips = tmpFolder / "cut_strips.tif";
    writeTestTiff( strips, { .size = { 4, 4 }, .rowsPerStrip = 1, .directoryFirst = true }, std::vector<uint8_t>( 16, 7 ) );
    cutOff( strips );
    EXPECT_TRUE( RasterLoad::infoFromTiff( strips ).has_value() );
    expectError( RasterLoad::fromTiff( strips ), "Error reading row" );
    expectError( ImageLoad::fromTiff( strips ), "Error reading pixels" );

    const auto tiles = tmpFolder / "cut_tiles.tif";
    writeTestTiff( tiles, { .size = { 20, 18 }, .tileSize = { 16, 16 }, .directoryFirst = true }, std::vector<uint8_t>( 20 * 18, 7 ) );
    cutOff( tiles );
    expectError( RasterLoad::fromTiff( tiles ), "Error reading tile" );

    // the data size does not match the dimensions
    Raster raster{
        .info = {
            .dims = { 2, 2, 1 },
            .type = ScalarType::UInt8,
        },
        .data = { 1, 2, 3 },
    };
    EXPECT_FALSE( RasterSave::toTiff( raster, tmpFolder / "wrong.tif" ).has_value() );

    // a TIFF file stores one layer
    raster.info.dims = { 2, 1, 2 };
    raster.data = { 1, 2, 3, 4 };
    EXPECT_FALSE( RasterSave::toTiff( raster, tmpFolder / "layers.tif" ).has_value() );

    raster.info = { .dims = { 1, 1, 1 }, .type = ScalarType::Float32_4 };
    raster.data.resize( raster.info.dataSize() );
    EXPECT_FALSE( RasterSave::toTiff( raster, tmpFolder / "float4.tif" ).has_value() );
}

TEST( MRMesh, TiffDistanceMapSave )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "dmap.tiff";

    DistanceMap dmap( 3, 2 );
    for ( int y = 0; y < 2; ++y )
        for ( int x = 0; x < 3; ++x )
            dmap.set( x, y, float( 10 * y + x ) );
    dmap.unset( 1, 1 );
    const AffineXf3f xf( Matrix3f::scale( 0.5f, -0.5f, 1.f ), Vector3f( 10.f, 20.f, 0.f ) );
    auto saveRes = DistanceMapSave::toTiff( dmap, path, { .xf = &xf } );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

    auto raster = RasterLoad::fromTiff( path );
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    EXPECT_EQ( raster->info, ( RasterInfo{ .dims = { 3, 2, 1 }, .type = ScalarType::Float32 } ) );
    std::vector<float> values( 6 );
    ASSERT_EQ( raster->data.size(), values.size() * sizeof( float ) );
    std::memcpy( values.data(), raster->data.data(), raster->data.size() );
    EXPECT_EQ( values, std::vector<float>( dmap.data(), dmap.data() + 6 ) );

    // the file is still read by the old reader, together with the GeoTIFF transformation
    DistanceMapToWorld toWorld;
    auto loaded = DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( std::vector<float>( loaded->data(), loaded->data() + 6 ), values );
    EXPECT_EQ( AffineXf3f( toWorld ), xf );
}

TEST( MRMesh, TiffCancel )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "strips.tif";
    writeTestTiff( path, { .size = { 4, 4 }, .rowsPerStrip = 1 }, std::vector<uint8_t>( 16, 7 ) );

    // the callback stops at the second report
    int calls = 0;
    const ProgressCallback cancelSecond = [&] ( float ) { return ++calls < 2; };
    auto loaded = RasterLoad::fromTiff( path, { .progress = cancelSecond } );
    ASSERT_FALSE( loaded.has_value() );
    // the file name is not appended, so that callers can recognize the cancellation
    EXPECT_EQ( loaded.error(), stringOperationCanceled() );

    calls = 0;
    Raster raster{ .info = { .dims = { 2, 4, 1 }, .type = ScalarType::UInt8 } };
    raster.data.resize( raster.info.dataSize() );
    auto saveRes = RasterSave::toTiff( raster, tmpFolder / "raster.tif", { .progress = cancelSecond } );
    ASSERT_FALSE( saveRes.has_value() );
    EXPECT_EQ( saveRes.error(), stringOperationCanceled() );
}
#endif //!MRIOEXTRAS_NO_TIFF

} //namespace MR

#endif //!__EMSCRIPTEN__ && !MRMESH_NO_TIFF
