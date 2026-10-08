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
    bool separatePlanes = false;
    // the image is stored in tiles of this size if it is not zero, otherwise in strips
    Vector2i tileSize;
    // the number of rows in a strip, zero means a single strip
    int rowsPerStrip = 0;
    std::vector<uint16_t> extraSamples;
    // 3 * 2^bitsPerSample values for a palette image
    std::vector<uint16_t> colorMap;
};

// writes an uncompressed TIFF file in the native byte order; `samples` holds the pixels in the stored order:
// rows one after another, the samples of a pixel together, each sample takes bitsPerSample / 8 bytes;
// the parts of the tiles outside the image are filled with the padding byte
static void writeTestTiff( const std::filesystem::path& path, const TestTiffLayout& layout, const std::vector<uint8_t>& samples, uint8_t padding = 0xFF )
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

    std::vector<uint8_t> file( 8 ); // the header is written in the end
    auto append = [&] ( const void* data, size_t size )
    {
        file.insert( file.end(), ( const uint8_t* )data, ( const uint8_t* )data + size );
    };

    std::vector<uint32_t> chunkOffsets, chunkSizes;
    for ( size_t plane = 0; plane < planeCount; ++plane )
    {
        for ( size_t y0 = 0; y0 < height; y0 += chunkHeight )
        {
            for ( size_t x0 = 0; x0 < width; x0 += chunkWidth )
            {
                chunkOffsets.push_back( uint32_t( file.size() ) );
                // the last strip ends with the last row, tiles are always complete
                const auto yEnd = tiled ? y0 + chunkHeight : std::min( y0 + chunkHeight, height );
                for ( size_t y = y0; y < yEnd; ++y )
                {
                    for ( size_t x = x0; x < x0 + chunkWidth; ++x )
                    {
                        for ( size_t s = 0; s < planeSamples; ++s )
                        {
                            if ( x < width && y < height )
                                append( samples.data() + ( ( y * width + x ) * samplesPerPixel + plane + s ) * sampleSize, sampleSize );
                            else
                                file.insert( file.end(), sampleSize, padding );
                        }
                    }
                }
                chunkSizes.push_back( uint32_t( file.size() - chunkOffsets.back() ) );
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
        { 259, SHORT, { 1 } }, // Compression: none
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

    // the image file directory starts on a word boundary, the values longer than 4 bytes are stored after it
    if ( file.size() % 2 != 0 )
        file.push_back( 0 );
    const auto directoryPos = uint32_t( file.size() );
    const auto externalPos = directoryPos + uint32_t( 2 + 12 * entries.size() + 4 );
    std::vector<uint8_t> external;
    auto write16 = [&] ( uint16_t v ) { append( &v, sizeof( v ) ); };
    auto write32 = [&] ( uint32_t v ) { append( &v, sizeof( v ) ); };
    write16( uint16_t( entries.size() ) );
    for ( const auto& entry : entries )
    {
        write16( entry.tag );
        write16( entry.type );
        write32( uint32_t( entry.values.size() ) );
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
        if ( bytes.size() <= 4 )
        {
            bytes.resize( 4, 0 );
            append( bytes.data(), bytes.size() );
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
    file.insert( file.end(), external.begin(), external.end() );

    // all numbers are written in the native byte order
    std::memcpy( file.data(), std::endian::native == std::endian::little ? "II" : "MM", 2 );
    const uint16_t magic = 42;
    std::memcpy( file.data() + 2, &magic, sizeof( magic ) );
    std::memcpy( file.data() + 4, &directoryPos, sizeof( directoryPos ) );

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
    // the padding is 0xFFFF
    writeTestTiff( path, { .size = size, .bitsPerSample = 16, .tileSize = tileSize }, toBytes( pixels ), 0xFF );

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

    // writeRawTiff marks the file with PHOTOMETRIC_MINISWHITE, which is ignored for floating-point samples
    auto raster = RasterLoad::fromTiff( path );
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    EXPECT_FALSE( raster->info.minIsWhite );

    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->resolution, Vector2i( 2, 2 ) );
    // the values are scaled to [0, 255], and the rows of Image go from bottom to top
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
    loaded = ImageLoad::fromTiff( savedPath );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->pixels, image.pixels );
}

TEST( MRMesh, TiffRasterRoundTrip )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    for ( auto type : { ScalarType::UInt8, ScalarType::Int8, ScalarType::UInt16, ScalarType::Int16, ScalarType::UInt32, ScalarType::Int32,
        ScalarType::UInt64, ScalarType::Int64, ScalarType::Float32, ScalarType::Float64 } )
    {
        for ( int channels = 1; channels <= 5; ++channels )
        {
            Raster raster{
                .info = {
                    .resolution = { 5, 3 },
                    .channels = channels,
                    .sampleType = type,
                },
            };
            raster.data.resize( raster.info.dataSize() );
            for ( size_t i = 0; i < raster.data.size(); ++i )
                raster.data[i] = uint8_t( i * 37 + 11 );

            const auto path = tmpFolder / ( "raster" + std::to_string( int( type ) ) + "_" + std::to_string( channels ) + ".tif" );
            auto saveRes = RasterSave::toAnySupportedFormat( raster, path );
            ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

            auto loaded = RasterLoad::fromAnySupportedFormat( path );
            ASSERT_TRUE( loaded.has_value() ) << loaded.error();
            EXPECT_EQ( loaded->info.resolution, raster.info.resolution );
            EXPECT_EQ( loaded->info.channels, channels );
            EXPECT_EQ( loaded->info.sampleType, type );
            EXPECT_TRUE( loaded->info.palette.empty() );
            EXPECT_FALSE( loaded->info.minIsWhite );
            EXPECT_FALSE( loaded->info.pixelToWorld.has_value() );
            EXPECT_FALSE( loaded->info.noData.has_value() );
            EXPECT_EQ( loaded->data, raster.data );

            auto info = RasterLoad::infoFromAnySupportedFormat( path );
            ASSERT_TRUE( info.has_value() ) << info.error();
            EXPECT_EQ( info->resolution, raster.info.resolution );
            EXPECT_EQ( info->channels, channels );
            EXPECT_EQ( info->sampleType, type );
        }
    }
}

TEST( MRMesh, TiffRasterMetadata )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "geo.tif";

    const AffineXf3f xf( Matrix3f::scale( 2.f, -3.f, 1.f ), Vector3f( 100.f, 200.f, 5.f ) );
    Raster raster{
        .info = {
            .resolution = { 2, 2 },
            .sampleType = ScalarType::Float32,
            .pixelToWorld = xf,
            .noData = -9999.,
        },
        .data = toBytes( std::vector<float>{ 1.f, 2.f, -9999.f, 4.f } ),
    };
    auto saveRes = RasterSave::toTiff( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    ASSERT_TRUE( loaded->info.pixelToWorld.has_value() );
    EXPECT_EQ( *loaded->info.pixelToWorld, xf );
    EXPECT_EQ( loaded->info.noData, -9999. );
    EXPECT_EQ( loaded->data, raster.data );

    raster.info.noData = std::nan( "" );
    saveRes = RasterSave::toTiff( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    auto info = RasterLoad::infoFromTiff( path );
    ASSERT_TRUE( info.has_value() ) << info.error();
    ASSERT_TRUE( info->noData.has_value() );
    EXPECT_TRUE( std::isnan( *info->noData ) );
}

TEST( MRMesh, TiffRasterPalette )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    for ( auto type : { ScalarType::UInt8, ScalarType::UInt16 } )
    {
        Raster raster{
            .info = {
                .resolution = { 3, 1 },
                .sampleType = type,
                .palette = { Color::red(), Color::green(), Color::blue() },
            },
        };
        raster.data = type == ScalarType::UInt8 ? std::vector<uint8_t>{ 2, 0, 1 } : toBytes( std::vector<uint16_t>{ 2, 0, 1 } );
        const auto path = tmpFolder / ( "palette" + std::to_string( int( type ) ) + ".tif" );
        auto saveRes = RasterSave::toTiff( raster, path );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

        auto loaded = RasterLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << loaded.error();
        EXPECT_EQ( loaded->info.sampleType, type );
        EXPECT_EQ( loaded->data, raster.data );
        // the palette has a color for each possible index
        ASSERT_EQ( loaded->info.palette.size(), type == ScalarType::UInt8 ? 256u : 65536u );
        EXPECT_EQ( std::vector<Color>( loaded->info.palette.begin(), loaded->info.palette.begin() + 3 ), raster.info.palette );

        auto image = ImageLoad::fromTiff( path );
        ASSERT_TRUE( image.has_value() ) << image.error();
        EXPECT_EQ( image->pixels, std::vector<Color>( { Color::blue(), Color::red(), Color::green() } ) );
    }

    // palette indices must be unsigned integers
    Raster floatRaster{
        .info = {
            .resolution = { 1, 1 },
            .sampleType = ScalarType::Float32,
            .palette = { Color::red() },
        },
        .data = toBytes( std::vector<float>{ 0.f } ),
    };
    EXPECT_FALSE( RasterSave::toTiff( floatRaster, tmpFolder / "palette_float.tif" ).has_value() );

    // an old-style color map with 8-bit values
    std::vector<uint16_t> colorMap( 3 * 256, 0 );
    colorMap[1] = 200; // red of index 1
    colorMap[256 + 1] = 100; // green of index 1
    colorMap[512 + 1] = 50; // blue of index 1
    const auto path = tmpFolder / "palette8.tif";
    writeTestTiff( path, {
        .size = { 2, 1 },
        .photometric = 3, // palette
        .colorMap = colorMap,
    }, { 1, 0 } );
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    ASSERT_EQ( loaded->info.palette.size(), 256u );
    EXPECT_EQ( loaded->info.palette[1], Color( 200, 100, 50 ) );
    EXPECT_EQ( loaded->info.palette[0], Color::black() );
}

TEST( MRMesh, TiffRasterMinIsWhite )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "white.tif";

    const Raster raster{
        .info = {
            .resolution = { 2, 1 },
            .sampleType = ScalarType::UInt8,
            .minIsWhite = true,
        },
        .data = { 0, 200 },
    };
    auto saveRes = RasterSave::toTiff( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_TRUE( loaded->info.minIsWhite );
    EXPECT_EQ( loaded->data, raster.data );

    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 255, 255 ), Color( 55, 55, 55 ) } ) );
}

TEST( MRMesh, TiffRasterStoredLayouts )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // RGB image with 16-bit samples
    const Vector2i size{ 5, 3 };
    std::vector<uint8_t> samples( size_t( size.x ) * size.y * 3 * sizeof( uint16_t ) );
    for ( size_t i = 0; i < samples.size(); ++i )
        samples[i] = uint8_t( i * 7 + 3 );
    const TestTiffLayout base{
        .size = size,
        .bitsPerSample = 16,
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
        EXPECT_EQ( loaded->info.resolution, size ) << c.name;
        EXPECT_EQ( loaded->info.channels, 3 ) << c.name;
        EXPECT_EQ( loaded->info.sampleType, ScalarType::UInt16 ) << c.name;
        EXPECT_EQ( loaded->data, samples ) << c.name;
    }
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
        const auto path = tmpFolder / ( "orientation" + std::to_string( orientation ) + ".tif" );
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
    EXPECT_EQ( loaded->info.channels, 4 );
    EXPECT_EQ( loaded->info.sampleType, ScalarType::UInt8 );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( {
        255, 255, 255, 255,
        0, 0, 0, 255,
        0, 255, 255, 255,
    } ) );
}

TEST( MRMesh, TiffRasterErrors )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    // libtiff's RGBA reader does not support 12-bit samples either
    const auto path12 = tmpFolder / "gray12.tif";
    writeTestTiff( path12, { .size = { 2, 1 }, .bitsPerSample = 12 }, { 1, 2 } );
    EXPECT_FALSE( RasterLoad::fromTiff( path12 ).has_value() );
    EXPECT_FALSE( RasterLoad::infoFromTiff( path12 ).has_value() );
    EXPECT_FALSE( ImageLoad::fromTiff( path12 ).has_value() );

    const auto notTiff = tmpFolder / "text.tif";
    std::ofstream( notTiff ) << "not a TIFF file";
    EXPECT_FALSE( RasterLoad::fromTiff( notTiff ).has_value() );

    // the data size does not match the resolution
    const Raster raster{
        .info = {
            .resolution = { 2, 2 },
            .sampleType = ScalarType::UInt8,
        },
        .data = { 1, 2, 3 },
    };
    EXPECT_FALSE( RasterSave::toTiff( raster, tmpFolder / "wrong.tif" ).has_value() );
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
    EXPECT_EQ( raster->info.resolution, Vector2i( 3, 2 ) );
    EXPECT_EQ( raster->info.channels, 1 );
    EXPECT_EQ( raster->info.sampleType, ScalarType::Float32 );
    EXPECT_FALSE( raster->info.minIsWhite );
    EXPECT_EQ( raster->info.pixelToWorld, xf );
    EXPECT_EQ( raster->info.noData, double( DistanceMap::NOT_VALID_VALUE ) );
    std::vector<float> values( 6 );
    ASSERT_EQ( raster->data.size(), values.size() * sizeof( float ) );
    std::memcpy( values.data(), raster->data.data(), raster->data.size() );
    EXPECT_EQ( values, std::vector<float>( dmap.data(), dmap.data() + 6 ) );

    // the file is still read by the old reader
    DistanceMapToWorld toWorld;
    auto loaded = DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( std::vector<float>( loaded->data(), loaded->data() + 6 ), values );
    EXPECT_EQ( AffineXf3f( toWorld ), xf );
}
#endif //!MRIOEXTRAS_NO_TIFF

} //namespace MR

#endif //!__EMSCRIPTEN__ && !MRMESH_NO_TIFF
