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
    uint16_t orientation = 1; // top-left
    uint16_t compression = 1; // none, the samples are written uncompressed anyway
    bool separatePlanes = false;
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
    const size_t planeCount = layout.separatePlanes ? samplesPerPixel : 1;
    const size_t planeSamples = layout.separatePlanes ? 1 : samplesPerPixel;

    // the pixels by chunks, their offsets are relative to the start of the pixels until the position of the pixels is known
    std::vector<uint8_t> pixels;
    std::vector<uint32_t> chunkOffsets, chunkSizes;
    for ( size_t plane = 0; plane < planeCount; ++plane )
    {
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
                        for ( size_t s = 0; s < planeSamples; ++s )
                        {
                            if ( x < width && y < height )
                            {
                                const auto* sample = samples.data() + ( ( y * width + x ) * samplesPerPixel + plane + s ) * sampleSize;
                                pixels.insert( pixels.end(), sample, sample + sampleSize );
                            }
                            else
                            {
                                pixels.insert( pixels.end(), sampleSize, uint8_t( 0xFF ) );
                            }
                        }
                    }
                }
                chunkSizes.push_back( uint32_t( pixels.size() - chunkOffsets.back() ) );
            }
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
        { 259, SHORT, { layout.compression } }, // Compression
        { 262, SHORT, { layout.photometric } }, // PhotometricInterpretation
        { 274, SHORT, { layout.orientation } }, // Orientation
        { 277, SHORT, { layout.samplesPerPixel } }, // SamplesPerPixel
        { 284, SHORT, { layout.separatePlanes ? 2u : 1u } }, // PlanarConfiguration
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

static std::vector<uint8_t> readTestFile( const std::filesystem::path& path )
{
    std::ifstream in( path, std::ios::binary );
    return { std::istreambuf_iterator<char>( in ), std::istreambuf_iterator<char>() };
}

// returns the position of the value of the first directory entry with given tag in a TIFF file in the native byte order, or 0
static size_t findTiffValue( const std::vector<uint8_t>& file, uint16_t tag )
{
    if ( file.size() < 8 || std::memcmp( file.data(), std::endian::native == std::endian::little ? "II" : "MM", 2 ) != 0 )
        return 0;
    uint32_t directoryPos = 0;
    std::memcpy( &directoryPos, file.data() + 4, sizeof( directoryPos ) );
    uint16_t entryCount = 0;
    if ( size_t( directoryPos ) + 2 > file.size() )
        return 0;
    std::memcpy( &entryCount, file.data() + directoryPos, sizeof( entryCount ) );
    for ( size_t i = 0; i < entryCount; ++i )
    {
        const auto entryPos = size_t( directoryPos ) + 2 + 12 * i;
        if ( entryPos + 12 > file.size() )
            return 0;
        uint16_t entryTag = 0;
        std::memcpy( &entryTag, file.data() + entryPos, sizeof( entryTag ) );
        if ( entryTag == tag )
            return entryPos + 8;
    }
    return 0;
}

// returns the first value of a SHORT entry stored in the entry itself
static std::optional<uint16_t> readTiffShort( const std::filesystem::path& path, uint16_t tag )
{
    const auto file = readTestFile( path );
    const auto pos = findTiffValue( file, tag );
    if ( !pos )
        return {};
    uint16_t res = 0;
    std::memcpy( &res, file.data() + pos, sizeof( res ) );
    return res;
}

// replaces the value of a LONG entry stored in the entry itself
static void setTiffLong( const std::filesystem::path& path, uint16_t tag, uint32_t value )
{
    auto file = readTestFile( path );
    const auto pos = findTiffValue( file, tag );
    ASSERT_NE( pos, 0u );
    std::memcpy( file.data() + pos, &value, sizeof( value ) );
    std::ofstream out( path, std::ios::binary );
    out.write( ( const char* )file.data(), std::streamsize( file.size() ) );
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

// unassociated alpha is kept as is, unlike in libtiff's RGBA reader, which premultiplies colors by it
TEST( MRMesh, TiffImageStraightAlpha )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    const auto path = tmpFolder / "alpha.tif";
    writeTestTiff( path, {
        .size = { 2, 1 },
        .samplesPerPixel = 4,
        .photometric = 2, // RGB
        .extraSamples = { 2 }, // unassociated alpha
    }, { 255, 0, 0, 128, 10, 20, 30, 0 } );
    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->pixels, std::vector<Color>( { Color( 255, 0, 0, 128 ), Color( 10, 20, 30, 0 ) } ) );

    const Image image{
        .pixels = { Color( 255, 0, 0, 128 ), Color( 1, 2, 3, 4 ), Color( 200, 100, 50, 0 ), Color( 7, 8, 9, 255 ) },
        .resolution = { 2, 2 },
    };
    const auto savedPath = tmpFolder / "saved.tif";
    auto saveRes = ImageSave::toTiff( image, savedPath );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    // the reader ignores the tags, so they are checked in the file
    EXPECT_EQ( readTiffShort( savedPath, 262 ), uint16_t( 2 ) ); // PhotometricInterpretation: RGB
    EXPECT_EQ( readTiffShort( savedPath, 338 ), uint16_t( 2 ) ); // ExtraSamples: unassociated alpha
    loaded = ImageLoad::fromTiff( savedPath );
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

TEST( MRMesh, TiffRasterColorConversions )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // 16-bit color samples are rounded to 8 bits
    const auto rgb16 = tmpFolder / "rgb16.tif";
    writeTestTiff( rgb16, { .size = { 1, 1 }, .bitsPerSample = 16, .samplesPerPixel = 3, .photometric = 2 },
        toBytes( std::vector<uint16_t>{ 0xFFFF, 0x8000, 0x0081 } ) );
    auto loaded = RasterLoad::fromTiff( rgb16 );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGB8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 255, 128, 1 } ) );

    // floating-point color samples are clamped to [0, 1], the samples after alpha are ignored
    const auto rgbaFloat = tmpFolder / "rgbaf.tif";
    writeTestTiff( rgbaFloat, {
        .size = { 1, 1 },
        .bitsPerSample = 32,
        .samplesPerPixel = 5,
        .sampleFormat = 3, // floating point
        .photometric = 2,
        .extraSamples = { 2, 0 },
    }, toBytes( std::vector<float>{ 2.f, 0.5f, -1.f, 1.f, 7.f } ) );
    loaded = RasterLoad::fromTiff( rgbaFloat );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGBA8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 255, 127, 0, 255 } ) );
}

// gray images with more samples per pixel are decoded by libtiff, which shows the second sample as alpha if ExtraSamples says so
TEST( MRMesh, TiffRasterGrayExtraSamples )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // libtiff declares a missing extra sample as unspecified
    const std::vector<uint16_t> extraSamples[] = { {}, { 0 }, { 2 } };
    const std::vector<uint8_t> expected[] = {
        { 10, 10, 10, 255, 20, 20, 20, 255 },
        { 10, 10, 10, 255, 20, 20, 20, 255 },
        { 10, 10, 10, 0, 20, 20, 20, 255 },
    };
    for ( size_t i = 0; i < std::size( extraSamples ); ++i )
    {
        const auto path = tmpFolder / fmt::format( "gray{}.tif", i );
        writeTestTiff( path, { .size = { 2, 1 }, .samplesPerPixel = 2, .extraSamples = extraSamples[i] }, { 10, 0, 20, 255 } );
        auto loaded = RasterLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << loaded.error();
        EXPECT_EQ( loaded->info.type, ScalarType::RGBA8 );
        EXPECT_EQ( loaded->data, expected[i] ) << i;
    }

    // libtiff cannot decode floating-point samples, so the first sample is kept
    const auto floatPath = tmpFolder / "gray_float.tif";
    writeTestTiff( floatPath, { .size = { 2, 1 }, .bitsPerSample = 32, .samplesPerPixel = 2, .sampleFormat = 3 },
        toBytes( std::vector<float>{ 1.5f, 0.f, -2.f, 0.f } ) );
    auto loaded = RasterLoad::fromTiff( floatPath );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::Float32 );
    EXPECT_EQ( loaded->data, toBytes( std::vector<float>{ 1.5f, -2.f } ) );
}

TEST( MRMesh, TiffRasterIntegerAndNaNColors )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // as in libtiff's RGBA reader, signed samples are taken as unsigned ones
    const auto rgb8 = tmpFolder / "rgb8s.tif";
    writeTestTiff( rgb8, { .size = { 1, 1 }, .samplesPerPixel = 3, .sampleFormat = 2, .photometric = 2 }, { 0xFF, 0x80, 0x01 } );
    auto loaded = RasterLoad::fromTiff( rgb8 );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGB8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 255, 128, 1 } ) );

    const auto rgb16 = tmpFolder / "rgb16s.tif";
    writeTestTiff( rgb16, { .size = { 1, 1 }, .bitsPerSample = 16, .samplesPerPixel = 3, .sampleFormat = 2, .photometric = 2 },
        toBytes( std::vector<int16_t>{ 20000, -1, 0 } ) );
    loaded = RasterLoad::fromTiff( rgb16 );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 78, 255, 0 } ) );

    // wider samples keep their high byte
    const auto rgb32 = tmpFolder / "rgb32.tif";
    writeTestTiff( rgb32, { .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 3, .photometric = 2 },
        toBytes( std::vector<uint32_t>{ 0xFF000000, 0x80FFFFFF, 0x00FFFFFF } ) );
    loaded = RasterLoad::fromTiff( rgb32 );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 255, 128, 0 } ) );

    // NaN becomes 0
    const auto rgbFloat = tmpFolder / "rgbf.tif";
    writeTestTiff( rgbFloat, { .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 3, .sampleFormat = 3, .photometric = 2 },
        toBytes( std::vector<float>{ std::nanf( "" ), 0.5f, 1.f } ) );
    loaded = RasterLoad::fromTiff( rgbFloat );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 0, 127, 255 } ) );
}

// the formats that libtiff cannot decode keep the stored values of the first sample, as the old image loader read them
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

// palette images are decoded by libtiff
TEST( MRMesh, TiffRasterPalette )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // red, green and blue colors for the indices 0, 1, 2, the rest is black
    std::vector<uint16_t> colorMap( 3 * 256, 0 );
    colorMap[0] = 0xFFFF;
    colorMap[256 + 1] = 0xFFFF;
    colorMap[512 + 2] = 0xFFFF;
    const auto path = tmpFolder / "palette.tif";
    writeTestTiff( path, { .size = { 3, 1 }, .photometric = 3, .colorMap = colorMap }, { 2, 0, 1 } );
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGBA8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255 } ) );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color::blue(), Color::red(), Color::green() } ) );

    // an old-style color map with 8-bit values
    std::fill( colorMap.begin(), colorMap.end(), uint16_t( 0 ) );
    colorMap[1] = 200; // red of index 1
    colorMap[256 + 1] = 100; // green of index 1
    colorMap[512 + 1] = 50; // blue of index 1
    const auto oldPath = tmpFolder / "palette_old.tif";
    writeTestTiff( oldPath, { .size = { 2, 1 }, .photometric = 3, .colorMap = colorMap }, { 1, 0 } );
    loaded = RasterLoad::fromTiff( oldPath );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 200, 100, 50, 255, 0, 0, 0, 255 } ) );
}

TEST( MRMesh, TiffRasterMinIsWhite )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // the raster keeps the stored gray values, the image inverts them as libtiff's RGBA reader does
    const auto gray = tmpFolder / "white.tif";
    writeTestTiff( gray, { .size = { 2, 1 }, .photometric = 0 }, { 0, 200 } );
    auto loaded = RasterLoad::fromTiff( gray );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::UInt8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 0, 200 } ) );
    auto image = ImageLoad::fromTiff( gray );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 255, 255 ), Color( 55, 55, 55 ) } ) );

    // gray with alpha is decoded by libtiff, which inverts the gray values
    const auto grayAlpha = tmpFolder / "white_alpha.tif";
    writeTestTiff( grayAlpha, { .size = { 2, 1 }, .samplesPerPixel = 2, .photometric = 0, .extraSamples = { 2 } }, { 0, 128, 200, 255 } );
    loaded = RasterLoad::fromTiff( grayAlpha );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGBA8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 255, 255, 255, 128, 55, 55, 55, 255 } ) );
    image = ImageLoad::fromTiff( grayAlpha );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 255, 255, 128 ), Color( 55, 55, 55, 255 ) } ) );
}

TEST( MRMesh, TiffRasterStoredLayouts )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // RGB image with 8-bit samples
    const Vector2i size{ 5, 3 };
    std::vector<uint8_t> samples( size_t( size.x ) * size.y * 3 );
    for ( size_t i = 0; i < samples.size(); ++i )
        samples[i] = uint8_t( i * 7 + 3 );
    const TestTiffLayout base{
        .size = size,
        .samplesPerPixel = 3,
        .photometric = 2, // RGB
    };

    struct Case
    {
        const char* name;
        bool separatePlanes = false;
        Vector2i tileSize;
        int rowsPerStrip = 0;
    };
    // the tiles of the last column and of the last row extend beyond the image
    const Case cases[] = {
        { .name = "strip" },
        { .name = "strips", .rowsPerStrip = 2 },
        { .name = "tiles", .tileSize = { 4, 2 } },
        { .name = "planeStrips", .separatePlanes = true, .rowsPerStrip = 2 },
        { .name = "planeTiles", .separatePlanes = true, .tileSize = { 4, 2 } },
    };
    for ( const auto& c : cases )
    {
        auto layout = base;
        layout.separatePlanes = c.separatePlanes;
        layout.tileSize = c.tileSize;
        layout.rowsPerStrip = c.rowsPerStrip;
        const auto path = tmpFolder / ( std::string( c.name ) + ".tif" );
        writeTestTiff( path, layout, samples );

        auto loaded = RasterLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << c.name << ": " << loaded.error();
        EXPECT_EQ( loaded->info, ( RasterInfo{ .dims = Vector3i( size.x, size.y, 1 ), .type = ScalarType::RGB8 } ) ) << c.name;
        EXPECT_EQ( loaded->data, samples ) << c.name;
    }

    // 16-bit gray values in tiles are kept as stored
    std::vector<uint16_t> gray( size_t( size.x ) * size.y );
    for ( size_t i = 0; i < gray.size(); ++i )
        gray[i] = uint16_t( 1000 * i + 7 );
    const auto grayPath = tmpFolder / "grayTiles.tif";
    writeTestTiff( grayPath, { .size = size, .bitsPerSample = 16, .tileSize = { 4, 2 } }, toBytes( gray ) );
    auto loaded = RasterLoad::fromTiff( grayPath );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::UInt16 );
    EXPECT_EQ( loaded->data, toBytes( gray ) );
}

TEST( MRMesh, TiffRasterOrientation )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // the stored rows: { 1, 2, 3 }, { 4, 5, 6 }
    const std::vector<uint8_t> stored{ 1, 2, 3, 4, 5, 6 };
    const std::vector<uint8_t> flipX{ 3, 2, 1, 6, 5, 4 };
    const std::vector<uint8_t> flipXY{ 6, 5, 4, 3, 2, 1 };
    const std::vector<uint8_t> flipY{ 4, 5, 6, 1, 2, 3 };
    // as in libtiff's RGBA reader, the orientations with swapped rows and columns are read without the swap
    const std::vector<uint8_t> expected[8] = { stored, flipX, flipXY, flipY, stored, flipX, flipXY, flipY };
    for ( uint16_t orientation = 1; orientation <= 8; ++orientation )
    {
        const auto path = tmpFolder / fmt::format( "orientation{}.tif", orientation );
        writeTestTiff( path, { .size = { 3, 2 }, .orientation = orientation }, stored );
        auto loaded = RasterLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << loaded.error();
        EXPECT_EQ( loaded->data, expected[orientation - 1] ) << orientation;
    }
}

// the formats without plain sample values are decoded by libtiff to 8-bit RGBA
TEST( MRMesh, TiffRasterDecoded )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "cmyk.tif";

    writeTestTiff( path, {
        .size = { 3, 1 },
        .samplesPerPixel = 4,
        .photometric = 5, // separated: CMYK
    }, {
        0, 0, 0, 0, // white
        0, 0, 0, 255, // black
        255, 0, 0, 0, // cyan
    } );
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info.type, ScalarType::RGBA8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( {
        255, 255, 255, 255,
        0, 0, 0, 255,
        0, 255, 255, 255,
    } ) );

    // libtiff applies the orientation itself: the bottom row goes first in the file
    const auto bottomLeft = tmpFolder / "cmyk_bottom_left.tif";
    writeTestTiff( bottomLeft, {
        .size = { 2, 2 },
        .samplesPerPixel = 4,
        .photometric = 5,
        .orientation = 4, // bottom-left
    }, {
        0, 0, 0, 0, 0, 0, 0, 255, // bottom row: white, black
        255, 0, 0, 0, 0, 0, 0, 0, // top row: cyan, white
    } );
    loaded = RasterLoad::fromTiff( bottomLeft );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( {
        0, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 0, 0, 0, 255,
    } ) );
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

    // libtiff reads a palette image without a color map, or with a color map of a wrong size, as a gray one
    const auto noColorMap = tmpFolder / "no_color_map.tif";
    writeTestTiff( noColorMap, { .size = { 2, 1 }, .photometric = 3 }, { 0, 200 } );
    auto gray = RasterLoad::fromTiff( noColorMap );
    ASSERT_TRUE( gray.has_value() ) << gray.error();
    EXPECT_EQ( gray->info.type, ScalarType::UInt8 );
    EXPECT_EQ( gray->data, std::vector<uint8_t>( { 0, 200 } ) );
    const auto shortColorMap = tmpFolder / "short_color_map.tif";
    writeTestTiff( shortColorMap, { .size = { 2, 1 }, .photometric = 3, .colorMap = std::vector<uint16_t>( 256, 0xFFFF ) }, { 0, 200 } );
    auto image = ImageLoad::fromTiff( shortColorMap );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 200, 200, 200 ) } ) );

    // the samples cannot be decoded without the codec
    const auto unknownCodec = tmpFolder / "unknown_codec.tif";
    writeTestTiff( unknownCodec, { .size = { 2, 1 }, .compression = 60000 }, { 0, 1 } );
    expectError( RasterLoad::infoFromTiff( unknownCodec ), "compression" );
    expectError( RasterLoad::fromTiff( unknownCodec ), "compression" );

    // the declared size does not fit in memory, although each side fits in int
    const auto tooLarge = tmpFolder / "too_large.tif";
    writeTestTiff( tooLarge, { .size = { 1, 1 } }, { 0 } );
    setTiffLong( tooLarge, 256, 0x7FFFFFFF ); // ImageWidth
    setTiffLong( tooLarge, 257, 0x7FFFFFFF ); // ImageLength
    setTiffLong( tooLarge, 278, 0xFFFFFFFF ); // RowsPerStrip: a single strip
    expectError( RasterLoad::fromTiff( tooLarge ), "too large" );
    expectError( ImageLoad::fromTiff( tooLarge ), "too large" );

    // the tile width does not fit in int
    const auto wideTiles = tmpFolder / "wide_tiles.tif";
    writeTestTiff( wideTiles, { .size = { 16, 16 }, .tileSize = { 16, 16 } }, std::vector<uint8_t>( 256 ) );
    setTiffLong( wideTiles, 322, 0x80000000 ); // TileWidth
    expectError( RasterLoad::fromTiff( wideTiles ), "tiles" );

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
    expectError( RasterLoad::fromTiff( strips ), "Error reading strip" );
    expectError( ImageLoad::fromTiff( strips ), "Error reading strip" );

    const auto tiles = tmpFolder / "cut_tiles.tif";
    writeTestTiff( tiles, { .size = { 20, 18 }, .tileSize = { 16, 16 }, .directoryFirst = true }, std::vector<uint8_t>( 20 * 18, 7 ) );
    cutOff( tiles );
    expectError( RasterLoad::fromTiff( tiles ), "Error reading tile" );

    const auto decoded = tmpFolder / "cut_cmyk.tif";
    writeTestTiff( decoded, { .size = { 2, 2 }, .samplesPerPixel = 4, .photometric = 5, .rowsPerStrip = 1, .directoryFirst = true },
        std::vector<uint8_t>( 16, 7 ) );
    cutOff( decoded );
    expectError( RasterLoad::fromTiff( decoded ), "Error reading pixels" );

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

    EXPECT_EQ( readTiffShort( path, 262 ), uint16_t( 1 ) ); // PhotometricInterpretation: BlackIsZero

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

    calls = 0;
    saveRes = DistanceMapSave::toTiff( DistanceMap( 2, 4 ), tmpFolder / "dmap.tiff", { .progress = cancelSecond } );
    ASSERT_FALSE( saveRes.has_value() );
    EXPECT_EQ( saveRes.error(), stringOperationCanceled() );
}
#endif //!MRIOEXTRAS_NO_TIFF

} //namespace MR

#endif //!__EMSCRIPTEN__ && !MRMESH_NO_TIFF
