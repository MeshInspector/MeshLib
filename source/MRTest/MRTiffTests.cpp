#include "MRIOExtras/MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRDistanceMapLoad.h"
#include "MRMesh/MRDistanceMapSave.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRRasterLoad.h"
#include "MRMesh/MRRasterSave.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRSpdlog.h"

#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace MR
{

// Cyrillic "test" and a Latin letter with diaeresis: no single ANSI code page has both
static const std::filesystem::path cNonAsciiName = u8"\u0442\u0435\u0441\u0442_\u00fc";

template <typename T>
static std::vector<uint8_t> toBytes( const std::vector<T>& values )
{
    std::vector<uint8_t> res( values.size() * sizeof( T ) );
    std::memcpy( res.data(), values.data(), res.size() );
    return res;
}

TEST( MRMesh, TiffNonAsciiPath )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto dir = tmpFolder / cNonAsciiName;
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::create_directory( dir, ec ) ) << systemToUtf8( ec.message() );
    const auto path = dir / "img.tif";

    const Raster raster{
        .info = { .dims = { 3, 2, 1 }, .type = ScalarType::Float32 },
        .data = toBytes( std::vector<float>{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f } ),
    };
    auto saveRes = RasterSave::toTiff( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    EXPECT_TRUE( std::filesystem::is_regular_file( path, ec ) );

    auto info = RasterLoad::infoFromTiff( path );
    ASSERT_TRUE( info.has_value() ) << info.error();
    EXPECT_EQ( *info, raster.info );

    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, raster.data );
}

// layout of a hand-written TIFF file
struct TestTiffLayout
{
    Vector2i size;
    uint16_t bitsPerSample = 8;
    uint16_t samplesPerPixel = 1;
    uint16_t sampleFormat = 1; // unsigned integer
    uint16_t photometric = 1; // BlackIsZero
    // the samples are written uncompressed whatever the tag says
    uint16_t compression = 1; // none
    uint16_t orientation = 1; // TopLeft
    // the image is stored in tiles of this size if it is not zero, otherwise in strips
    Vector2i tileSize;
    // the number of rows in a strip, zero means a single strip
    int rowsPerStrip = 0;
    std::vector<uint16_t> extraSamples;
    // 3 * 2^bitsPerSample values for a palette image
    std::vector<uint16_t> colorMap;
    // tags with DOUBLE values
    std::vector<std::pair<int, std::vector<double>>> doubleTags;
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

    constexpr uint16_t SHORT = 3, LONG = 4, DOUBLE = 12;
    struct Entry
    {
        uint16_t tag;
        uint16_t type;
        std::vector<uint32_t> values;
        // the values of a DOUBLE entry
        std::vector<double> doubles;
    };
    std::vector<Entry> entries{
        { 256, LONG, { uint32_t( width ) } }, // ImageWidth
        { 257, LONG, { uint32_t( height ) } }, // ImageLength
        { 258, SHORT, std::vector<uint32_t>( samplesPerPixel, layout.bitsPerSample ) }, // BitsPerSample
        { 259, SHORT, { layout.compression } }, // Compression
        { 262, SHORT, { layout.photometric } }, // PhotometricInterpretation
        { 274, SHORT, { layout.orientation } }, // Orientation
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
    for ( const auto& [tag, doubles] : layout.doubleTags )
        entries.push_back( { uint16_t( tag ), DOUBLE, {}, doubles } );
    std::sort( entries.begin(), entries.end(), [] ( const Entry& a, const Entry& b ) { return a.tag < b.tag; } );

    auto valueBytes = [&] ( const Entry& entry )
    {
        // filled in place, as the file below: inserts into a vector give a false -Wstringop-overread in GCC 11 Release builds
        std::vector<uint8_t> bytes( entry.values.size() * ( entry.type == SHORT ? sizeof( uint16_t ) : sizeof( uint32_t ) )
            + entry.doubles.size() * sizeof( double ) );
        auto out = bytes.begin();
        for ( auto v : entry.values )
        {
            if ( entry.type == SHORT )
            {
                const auto s = uint16_t( v );
                out = std::copy( ( const uint8_t* )&s, ( const uint8_t* )&s + sizeof( s ), out );
            }
            else
            {
                out = std::copy( ( const uint8_t* )&v, ( const uint8_t* )&v + sizeof( v ), out );
            }
        }
        std::copy( ( const uint8_t* )entry.doubles.data(), ( const uint8_t* )( entry.doubles.data() + entry.doubles.size() ), out );
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
        write32( uint32_t( entry.values.size() + entry.doubles.size() ) );
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
    const auto& first = layout.directoryFirst ? directory : pixels;
    const auto& second = layout.directoryFirst ? pixels : directory;
    const auto secondPos = layout.directoryFirst ? pixelsPos : directoryPos;
    std::vector<uint8_t> file( secondPos + second.size() );
    std::memcpy( file.data(), std::endian::native == std::endian::little ? "II" : "MM", 2 );
    const uint16_t magic = 42;
    std::memcpy( file.data() + 2, &magic, sizeof( magic ) );
    const auto directoryOffset = uint32_t( directoryPos );
    std::memcpy( file.data() + 4, &directoryOffset, sizeof( directoryOffset ) );
    std::copy( first.begin(), first.end(), file.begin() + 8 );
    std::copy( second.begin(), second.end(), file.begin() + secondPos );

    std::ofstream out( path, std::ios::binary );
    out.write( ( const char* )file.data(), std::streamsize( file.size() ) );
}

// returns whether a TIFF file in the native byte order has the directory entry of a SHORT tag with one value
static bool hasTiffTag( const std::filesystem::path& path, uint16_t tag, uint16_t value )
{
    // the tag, its type SHORT, the count of values and the value padded to 4 bytes
    const struct { uint16_t tag, type; uint32_t count; uint16_t value, padding; } entry{ tag, 3, 1, value, 0 };
    std::ifstream in( path, std::ios::binary );
    const std::string file( ( std::istreambuf_iterator<char>( in ) ), std::istreambuf_iterator<char>() );
    return file.find( std::string( ( const char* )&entry, sizeof( entry ) ) ) != std::string::npos;
}

TEST( MRMesh, TiffTiledPartialTiles )
{
    // the tiles in the last column and in the last row extend beyond the image
    const Vector2i size{ 20, 18 };
    std::vector<uint16_t> pixels( size_t( size.x ) * size.y );
    for ( int y = 0; y < size.y; ++y )
        for ( int x = 0; x < size.x; ++x )
            pixels[x + y * size.x] = uint16_t( 100 * y + x );

    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "tiled.tif";
    // the tiles are padded with 0xFFFF
    writeTestTiff( path, { .size = size, .bitsPerSample = 16, .tileSize = { 16, 16 } }, toBytes( pixels ) );

    auto raster = RasterLoad::fromTiff( path );
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    EXPECT_EQ( raster->info, ( RasterInfo{ .dims = { size.x, size.y, 1 }, .type = ScalarType::UInt16 } ) );
    EXPECT_EQ( raster->data, toBytes( pixels ) );

    std::vector<float> floats;
    auto res = convertRasterToFloats( *raster, floats );
    ASSERT_TRUE( res.has_value() ) << res.error();
    EXPECT_EQ( floats, std::vector<float>( pixels.begin(), pixels.end() ) );
    // the caller's memory must have room for all values, and no more
    res = convertRasterToFloats( *raster, floats.data(), floats.size() - 1 );
    EXPECT_FALSE( res.has_value() );
}

// libtiff passes the number of values of the GeoTIFF tags as a 16-bit or a 32-bit value, depending on how the tags are registered:
// e.g. libgeotiff's tag extender registers them with a 16-bit count
TEST( MRMesh, TiffGeoTransform )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "geo.tif";

    // a raster of any type keeps the transformation, which distance maps read too
    const Raster raster{
        .info = { .dims = { 3, 2, 1 }, .type = ScalarType::UInt16 },
        .data = toBytes( std::vector<uint16_t>{ 0, 1, 2, 3, 4, 5 } ),
    };
    const AffineXf3f xf( Matrix3f::scale( 0.5f, -0.5f, 2.f ), Vector3f( 10.f, 20.f, 3.f ) );
    TiffSaveSettings settings;
    settings.xf = &xf;
    settings.noData = -9999.;
    auto saveRes = RasterSave::toTiff( raster, path, settings );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

    AffineXf3f loadedXf;
    auto loaded = RasterLoad::fromAnySupportedFormat( path, { .xf = &loadedXf } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->data, raster.data );
    EXPECT_EQ( loadedXf, xf );
    DistanceMapToWorld toWorld;
    ASSERT_TRUE( DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } ).has_value() );
    EXPECT_EQ( AffineXf3f( toWorld ), xf );
    std::optional<double> noData;
    TiffLoadSettings tiffSettings;
    tiffSettings.noData = &noData;
    ASSERT_TRUE( RasterLoad::fromTiff( path, tiffSettings ).has_value() );
    EXPECT_EQ( noData, -9999. );

    // a file without the transformation gives the identity, and no value meaning no data
    saveRes = RasterSave::toAnySupportedFormat( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    loaded = RasterLoad::fromAnySupportedFormat( path, { .xf = &loadedXf } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loadedXf, AffineXf3f() );
    ASSERT_TRUE( RasterLoad::fromTiff( path, tiffSettings ).has_value() );
    EXPECT_EQ( noData, std::nullopt );
}

// without the transformation matrix, the GeoTIFF transformation is made of a tie point and the pixel scale
TEST( MRMesh, TiffGeoTiePoint )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "tiePoint.tif";

    const std::vector<float> values{ 0.f, 1.f, 2.f, 3.f, 4.f, 5.f };
    writeTestTiff( path, { .size = { 3, 2 }, .bitsPerSample = 32, .sampleFormat = 3, .doubleTags = {
        { 33550, { 0.5, 0.25, 0. } }, // ModelPixelScaleTag: the zero scale of values keeps them as they are
        { 33922, { 0., 0., 0., 10., 20., 0. } }, // ModelTiePointTag: the raster point (0, 0, 0) is at (10, 20, 0)
    } }, toBytes( values ) );

    DistanceMapToWorld toWorld;
    auto loaded = DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( std::vector<float>( loaded->data(), loaded->data() + 6 ), values );
    // the rows go down while y goes up
    const AffineXf3f expected( Matrix3f::scale( 0.5f, -0.25f, 1.f ), Vector3f( 10.f, 20.f, 0.f ) );
    EXPECT_EQ( AffineXf3f( toWorld ), expected );
    AffineXf3f rasterXf;
    ASSERT_TRUE( RasterLoad::fromTiff( path, { .xf = &rasterXf } ).has_value() );
    EXPECT_EQ( rasterXf, expected );
}

// the tie point can be any raster point (I, J, K), the translation is the model point of the raster origin
TEST( MRMesh, TiffGeoTiePointNotAtOrigin )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "tiePoint.tif";

    struct GeoTags
    {
        std::vector<double> scale;
        std::vector<double> tiePoint;
        AffineXf3f expected;
    };
    const GeoTags cases[] = {
        // the raster point (1, 2, 0) is at (10, 20, 0): the origin is one column to the left and two rows up
        { { 0.5, 0.25, 0. }, { 1., 2., 0., 10., 20., 0. }, AffineXf3f( Matrix3f::scale( 0.5f, -0.25f, 1.f ), Vector3f( 9.5f, 20.5f, 0.f ) ) },
        // the raster point (1, 2, 4) is at (10, 20, 30), and the values are scaled too
        { { 0.5, 0.25, 2. }, { 1., 2., 4., 10., 20., 30. }, AffineXf3f( Matrix3f::scale( 0.5f, -0.25f, 2.f ), Vector3f( 9.5f, 20.5f, 22.f ) ) },
        // the zero scale of values keeps them as they are, whatever the tie point's K and Z
        { { 0.5, 0.25, 0. }, { 1., 2., 4., 10., 20., 30. }, AffineXf3f( Matrix3f::scale( 0.5f, -0.25f, 1.f ), Vector3f( 9.5f, 20.5f, 0.f ) ) },
    };
    for ( const auto& c : cases )
    {
        writeTestTiff( path, { .size = { 3, 2 }, .bitsPerSample = 32, .sampleFormat = 3, .doubleTags = {
            { 33550, c.scale }, // ModelPixelScaleTag
            { 33922, c.tiePoint }, // ModelTiePointTag
        } }, toBytes( std::vector<float>( 6 ) ) );

        DistanceMapToWorld toWorld;
        ASSERT_TRUE( DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } ).has_value() );
        EXPECT_EQ( AffineXf3f( toWorld ), c.expected );
        AffineXf3f rasterXf;
        ASSERT_TRUE( RasterLoad::fromTiff( path, { .xf = &rasterXf } ).has_value() );
        EXPECT_EQ( rasterXf, c.expected );
    }
}

// the colors that libtiff decodes give their luma as floats
TEST( MRMesh, TiffRasterDecodedLuma )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "cmyk.tif";
    // white, black and cyan
    writeTestTiff( path, { .size = { 3, 1 }, .samplesPerPixel = 4, .photometric = 5 }, { 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0 } );

    auto raster = RasterLoad::fromTiff( path );
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    std::vector<float> floats;
    auto res = convertRasterToFloats( *raster, floats );
    ASSERT_TRUE( res.has_value() ) << res.error();
    ASSERT_EQ( floats.size(), 3u );
    EXPECT_NEAR( floats[0], 255.f, 1e-3f );
    EXPECT_EQ( floats[1], 0.f );
    EXPECT_NEAR( floats[2], ( 0.587f + 0.114f ) * 255.f, 1e-3f );

    auto dmap = DistanceMapLoad::fromTiff( path );
    ASSERT_TRUE( dmap.has_value() ) << dmap.error();
    EXPECT_EQ( std::vector<float>( dmap->data(), dmap->data() + 3 ), floats );
}

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

    const Raster raster{
        .info = { .dims = { 2, 2, 1 }, .type = ScalarType::Float32 },
        .data = toBytes( std::vector<float>{
            0.f, 1.f, // top row
            2.f, 3.f, // bottom row
        } ),
    };
    auto saveRes = RasterSave::toTiff( raster, path );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();

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

// colors with alpha are loaded as saved: the alpha is unassociated, the colors are not multiplied by it
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
    EXPECT_TRUE( hasTiffTag( path, 338, 2 ) ); // ExtraSamples: unassociated alpha
    auto loaded = ImageLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->pixels, image.pixels );

    // the file starts from the top row, which is the last row of the image
    auto raster = RasterLoad::fromTiff( path );
    ASSERT_TRUE( raster.has_value() ) << raster.error();
    EXPECT_EQ( raster->info, ( RasterInfo{ .dims = { 2, 2, 1 }, .type = ScalarType::UInt8, .channels = 4 } ) );
    std::vector<Color> colors( 4 );
    ASSERT_EQ( raster->data.size(), colors.size() * sizeof( Color ) );
    std::memcpy( (void*)colors.data(), raster->data.data(), raster->data.size() );
    EXPECT_EQ( colors, std::vector<Color>( { image.pixels[2], image.pixels[3], image.pixels[0], image.pixels[1] } ) );

    // other files with unassociated alpha keep their colors too, which libtiff's decoder would multiply by the alpha,
    // and so do the files without ExtraSamples, as earlier versions saved them
    for ( const auto& extraSamples : { std::vector<uint16_t>{ 2 }, std::vector<uint16_t>{} } )
    {
        writeTestTiff( path, { .size = { 2, 1 }, .samplesPerPixel = 4, .photometric = 2, .extraSamples = extraSamples }, { 200, 100, 50, 128, 1, 2, 3, 0 } );
        loaded = ImageLoad::fromTiff( path );
        ASSERT_TRUE( loaded.has_value() ) << loaded.error();
        EXPECT_EQ( loaded->pixels, std::vector<Color>( { Color( 200, 100, 50, 128 ), Color( 1, 2, 3, 0 ) } ) ) << extraSamples.size() << " extra samples";
    }

    const Image wrongSize{ .pixels = { Color::red() }, .resolution = { 2, 2 } };
    EXPECT_FALSE( ImageSave::toTiff( wrongSize, path ).has_value() );
}

TEST( MRMesh, TiffRasterRoundTrip )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );

    for ( auto type : { ScalarType::UInt8, ScalarType::Int8, ScalarType::UInt16, ScalarType::Int16, ScalarType::UInt32, ScalarType::Int32,
        ScalarType::UInt64, ScalarType::Int64, ScalarType::Float32, ScalarType::Float64 } )
    for ( int channels : { 1, 3, 4 } )
    {
        Raster raster{
            .info = {
                .dims = { 5, 3, 1 },
                .type = type,
                .channels = channels,
            },
        };
        raster.data.resize( raster.info.dataSize() );
        for ( size_t i = 0; i < raster.data.size(); ++i )
            raster.data[i] = uint8_t( i * 37 + 11 );

        const auto path = tmpFolder / fmt::format( "raster{}_{}.tif", int( type ), channels );
        auto saveRes = RasterSave::toAnySupportedFormat( raster, path );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
        EXPECT_EQ( hasTiffTag( path, 338, 2 ), channels == 4 ); // ExtraSamples: unassociated alpha

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
    int channels;
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
        EXPECT_EQ( loaded->info, ( RasterInfo{ .dims = Vector3i( c.layout.size.x, c.layout.size.y, 1 ), .type = c.type, .channels = c.channels } ) ) << c.name;
        EXPECT_EQ( loaded->data, c.expected ) << c.name;
    }
}

// the color samples of any type are kept as stored, ImageLoad converts the ones that libtiff cannot decode
TEST( MRMesh, TiffRasterStoredColors )
{
    const auto rgb16 = toBytes( std::vector<uint16_t>{ 0xFFFF, 0x80FF, 0x00FF } );
    const auto rgbaf = toBytes( std::vector<float>{ 2.f, 0.5f, std::nanf( "" ), 1.f, 7.f } );
    const TestTiffLayout rgbafLayout{ .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 5, .sampleFormat = 3, .photometric = 2, .extraSamples = { 2, 0 } };
    checkTestTiffCases( {
        { "rgb16", { .size = { 1, 1 }, .bitsPerSample = 16, .samplesPerPixel = 3, .photometric = 2 }, rgb16, ScalarType::UInt16, 3, rgb16 },
        { "rgbaf", rgbafLayout, rgbaf, ScalarType::Float32, 5, rgbaf },
    } );

    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "rgbaf.tif";
    writeTestTiff( path, rgbafLayout, rgbaf );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 255, 127, 0, 255 ) } ) );
}

// the formats without plain samples are decoded by libtiff to RGBA colors
TEST( MRMesh, TiffRasterDecoded )
{
    checkTestTiffCases( {
        // white, black and cyan
        { "cmyk", { .size = { 3, 1 }, .samplesPerPixel = 4, .photometric = 5 }, { 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0 },
            ScalarType::UInt8, 4, { 255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255 } },
        { "grayAlpha", { .size = { 2, 1 }, .samplesPerPixel = 2, .extraSamples = { 2 } }, { 10, 0, 20, 255 },
            ScalarType::UInt8, 4, { 10, 10, 10, 0, 20, 20, 20, 255 } },
    } );
}

// the rows of the raster keep the stored order whatever the Orientation tag says, also in the formats decoded by libtiff;
// ImageLoad follows the tag
TEST( MRMesh, TiffRasterOrientation )
{
    // the first stored row is the bottom one
    checkTestTiffCases( {
        { "gray", { .size = { 1, 2 }, .orientation = 4 }, { 10, 20 }, ScalarType::UInt8, 1, { 10, 20 } },
        { "grayAlpha", { .size = { 1, 2 }, .samplesPerPixel = 2, .orientation = 4, .extraSamples = { 2 } }, { 10, 255, 20, 255 },
            ScalarType::UInt8, 4, { 10, 10, 10, 255, 20, 20, 20, 255 } },
    } );

    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "bottomUp.tif";
    writeTestTiff( path, { .size = { 1, 2 }, .orientation = 4 }, { 10, 20 } );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    // the rows of Image go from bottom to top too
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 10, 10, 10 ), Color( 20, 20, 20 ) } ) );
}

// the raster keeps the color indices, libtiff decodes the colors in the image
TEST( MRMesh, TiffRasterPalette )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "palette.tif";

    // red, green and blue colors for the indices 0, 1, 2, the rest is black
    std::vector<uint16_t> colorMap( 3 * 256, 0 );
    colorMap[0] = colorMap[256 + 1] = colorMap[512 + 2] = 0xFFFF;
    writeTestTiff( path, { .size = { 3, 1 }, .photometric = 3, .colorMap = colorMap }, { 2, 0, 1 } );
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info, ( RasterInfo{ .dims = { 3, 1, 1 }, .type = ScalarType::UInt8 } ) );
    EXPECT_EQ( loaded->data, std::vector<uint8_t>( { 2, 0, 1 } ) );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color::blue(), Color::red(), Color::green() } ) );
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
    // the gray images with several samples per pixel keep them all
    const auto bands = toBytes( std::vector<float>{ 1.5f, -2.f, 1e6f } );
    checkTestTiffCases( {
        { "strip", { .size = size, .samplesPerPixel = 3, .photometric = 2 }, rgb, ScalarType::UInt8, 3, rgb },
        { "strips", { .size = size, .samplesPerPixel = 3, .photometric = 2, .rowsPerStrip = 2 }, rgb, ScalarType::UInt8, 3, rgb },
        { "tiles", { .size = size, .samplesPerPixel = 3, .photometric = 2, .tileSize = { 4, 2 } }, rgb, ScalarType::UInt8, 3, rgb },
        { "grayTiles", { .size = size, .bitsPerSample = 16, .tileSize = { 4, 2 } }, toBytes( gray ), ScalarType::UInt16, 1, toBytes( gray ) },
        { "bands", { .size = { 1, 1 }, .bitsPerSample = 32, .samplesPerPixel = 3, .sampleFormat = 3 }, bands, ScalarType::Float32, 3, bands },
    } );
}

// the formats that libtiff cannot decode are read as stored, and their gray values are scaled in the image, as the old image loader did
TEST( MRMesh, TiffRasterUndecodedFormat )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "linearraw.tif";

    writeTestTiff( path, { .size = { 2, 1 }, .bitsPerSample = 16, .photometric = 34892 }, toBytes( std::vector<uint16_t>{ 0x0100, 0xFF00 } ) ); // LinearRaw
    auto loaded = RasterLoad::fromTiff( path );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->info, ( RasterInfo{ .dims = { 2, 1, 1 }, .type = ScalarType::UInt16 } ) );
    EXPECT_EQ( loaded->data, toBytes( std::vector<uint16_t>{ 0x0100, 0xFF00 } ) );
    auto image = ImageLoad::fromTiff( path );
    ASSERT_TRUE( image.has_value() ) << image.error();
    EXPECT_EQ( image->pixels, std::vector<Color>( { Color( 0, 0, 0 ), Color( 255, 255, 255 ) } ) );
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

    // libtiff opens a file compressed by an unknown method, but cannot decode it
    const auto unknownCompression = tmpFolder / "compression.tif";
    writeTestTiff( unknownCompression, { .size = { 2, 1 }, .compression = 60000 }, { 1, 2 } );
    expectError( RasterLoad::fromTiff( unknownCompression ), "Unsupported compression method: 60000" );
    expectError( RasterLoad::infoFromTiff( unknownCompression ), "Unsupported compression method" );
    expectError( ImageLoad::fromTiff( unknownCompression ), "Unsupported compression method" );

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

    // gray with alpha is not saved
    raster.info = { .dims = { 1, 1, 1 }, .type = ScalarType::UInt8, .channels = 2 };
    raster.data.resize( raster.info.dataSize() );
    EXPECT_FALSE( RasterSave::toTiff( raster, tmpFolder / "grayAlpha.tif" ).has_value() );
}

// libtiff's warnings and errors go to the log instead of stderr
TEST( MRMesh, TiffMessagesToLog )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    // libtiff warns about the fourth sample of an RGB image without ExtraSamples when it reads the file
    const auto rgba = tmpFolder / "rgba.tif";
    writeTestTiff( rgba, { .size = { 1, 1 }, .samplesPerPixel = 4, .photometric = 2 }, { 255, 0, 0, 255 } );
    const auto notTiff = tmpFolder / "text.tif";
    std::ofstream( notTiff ) << "not a TIFF file";

    std::ostringstream log;
    const auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>( log );
    sink->set_pattern( "%l %v" );
    Logger::instance().addSink( sink );
    EXPECT_TRUE( ImageLoad::fromTiff( rgba ).has_value() );
    EXPECT_FALSE( ImageLoad::fromTiff( notTiff ).has_value() );
    Logger::instance().removeSink( sink );

    EXPECT_NE( log.str().find( "warning libtiff: TIFFReadDirectory: Sum of Photometric" ), std::string::npos ) << log.str();
    EXPECT_NE( log.str().find( "error libtiff: " ), std::string::npos ) << log.str();
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

    // the distance map is read back together with the GeoTIFF transformation
    DistanceMapToWorld toWorld;
    auto loaded = DistanceMapLoad::fromTiff( path, { .distanceMapToWorld = &toWorld } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( std::vector<float>( loaded->data(), loaded->data() + 6 ), values );
    EXPECT_EQ( AffineXf3f( toWorld ), xf );

    // the saver is registered for .tif as well as .tiff, as the loader is
    const auto tifPath = tmpFolder / "dmap.tif";
    saveRes = DistanceMapSave::toAnySupportedFormat( dmap, tifPath, { .xf = &xf } );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    DistanceMapToWorld tifToWorld;
    loaded = DistanceMapLoad::fromAnySupportedFormat( tifPath, { .distanceMapToWorld = &tifToWorld } );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( std::vector<float>( loaded->data(), loaded->data() + 6 ), values );
    EXPECT_EQ( AffineXf3f( tifToWorld ), xf );
}

// the values of one channel equal to the value meaning no data become invalid in distance maps, NaN too
TEST( MRMesh, TiffDistanceMapNoData )
{
    UniqueTemporaryFolder tmpFolder;
    ASSERT_TRUE( tmpFolder );
    const auto path = tmpFolder / "dem.tif";

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto floats = toBytes( std::vector<float>{ 1.f, -9999.f, nan, 4.f } );
    struct NoDataCase
    {
        Raster raster;
        double noData;
        std::vector<bool> valid;
    };
    const NoDataCase cases[] = {
        { { .info = { .dims = { 4, 1, 1 }, .type = ScalarType::Float32 }, .data = floats }, -9999., { true, false, true, true } },
        { { .info = { .dims = { 4, 1, 1 }, .type = ScalarType::Float32 }, .data = floats }, nan, { true, true, false, true } },
        // integer samples are converted to floats first
        { { .info = { .dims = { 2, 1, 1 }, .type = ScalarType::Int16 }, .data = toBytes( std::vector<int16_t>{ 1, -32768 } ) }, -32768., { true, false } },
        // the value refers to the samples, not to the luma of colors
        { { .info = { .dims = { 2, 1, 1 }, .type = ScalarType::UInt8, .channels = 3 }, .data = { 0, 0, 0, 10, 20, 30 } }, 0., { true, true } },
    };
    for ( const auto& c : cases )
    {
        TiffSaveSettings settings;
        settings.noData = c.noData;
        auto saveRes = RasterSave::toTiff( c.raster, path, settings );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
        auto dmap = DistanceMapLoad::fromTiff( path );
        ASSERT_TRUE( dmap.has_value() ) << dmap.error();
        for ( size_t i = 0; i < c.valid.size(); ++i )
            EXPECT_EQ( dmap->isValid( i ), c.valid[i] ) << i;
    }
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

} //namespace MR

#endif //!MRIOEXTRAS_NO_TIFF
