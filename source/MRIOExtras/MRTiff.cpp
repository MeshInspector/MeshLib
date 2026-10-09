#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRFmt.h"

#include <tiffio.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <memory>

namespace
{
using namespace MR;

// GeoTIFF tag: http://geotiff.maptools.org/spec/geotiff2.6.html
constexpr uint32_t cModelTransformationTag = 34264;
// no-data value: https://gdal.org/en/stable/drivers/raster/gtiff.html#nodata-value
constexpr uint32_t cGdalNoDataTag = 42113;

struct TiffCloser
{
    void operator()( TIFF* tiff ) const { TIFFClose( tiff ); }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// on Windows, narrow TIFFOpen interprets the file name in the ANSI code page, which fails for non-ASCII names
TiffPtr openTiff( const std::filesystem::path& path, const char* mode )
{
#ifdef _WIN32
    return TiffPtr( TIFFOpenW( path.wstring().c_str(), mode ) );
#else
    return TiffPtr( TIFFOpen( utf8string( path ).c_str(), mode ) );
#endif
}

// calls f( i ) for each i in [0, count) in parallel by blocks
template <typename F>
void parallelForEach( size_t count, F&& f )
{
    constexpr size_t cBlockSize = 4096;
    ParallelFor( size_t( 0 ), chunkCount( count, cBlockSize ), [&] ( size_t block )
    {
        const auto end = std::min( count, ( block + 1 ) * cBlockSize );
        for ( auto i = block * cBlockSize; i < end; ++i )
            f( i );
    } );
}

// the value types that are stored as single samples
struct SampleType
{
    ScalarType type;
    uint16_t format;
    uint16_t bits;
};
constexpr SampleType cSampleTypes[] = {
    { ScalarType::UInt8, SAMPLEFORMAT_UINT, 8 },
    { ScalarType::Int8, SAMPLEFORMAT_INT, 8 },
    { ScalarType::UInt16, SAMPLEFORMAT_UINT, 16 },
    { ScalarType::Int16, SAMPLEFORMAT_INT, 16 },
    { ScalarType::UInt32, SAMPLEFORMAT_UINT, 32 },
    { ScalarType::Int32, SAMPLEFORMAT_INT, 32 },
    { ScalarType::UInt64, SAMPLEFORMAT_UINT, 64 },
    { ScalarType::Int64, SAMPLEFORMAT_INT, 64 },
    { ScalarType::Float32, SAMPLEFORMAT_IEEEFP, 32 },
    { ScalarType::Float64, SAMPLEFORMAT_IEEEFP, 64 },
};

// returns the type of the stored samples, or ScalarType::Unknown if it is not supported
ScalarType getSampleType( uint16_t sampleFormat, uint16_t bitsPerSample )
{
    // the samples of unspecified format are taken as unsigned integers
    if ( sampleFormat == SAMPLEFORMAT_VOID )
        sampleFormat = SAMPLEFORMAT_UINT;
    for ( const auto& s : cSampleTypes )
        if ( s.format == sampleFormat && s.bits == bitsPerSample )
            return s.type;
    return ScalarType::Unknown;
}

// how the pixels are stored in the file and how they become the values of the raster
struct TiffLayout
{
    Vector2i size;
    uint16_t samplesPerPixel = 1;
    uint16_t bitsPerSample = 1;
    uint16_t photometric = PHOTOMETRIC_MINISBLACK;
    uint16_t planarConfig = PLANARCONFIG_CONTIG;
    uint16_t orientation = ORIENTATION_TOPLEFT;
    std::optional<Vector2i> tileSize;
    // type of the samples if they are read as stored; ScalarType::Unknown if libtiff decodes the pixels to RGBA8 values
    ScalarType sampleType = ScalarType::Unknown;
    // type of the raster values: sampleType if the first sample of each pixel is taken as is,
    // otherwise RGB8 or RGBA8 made of the first three or four samples, or decoded by libtiff
    ScalarType valueType = ScalarType::RGBA8;

    RasterInfo rasterInfo() const
    {
        return { .dims = Vector3i( size.x, size.y, 1 ), .type = valueType };
    }
};

Expected<TiffLayout> readLayout( TIFF* tiff )
{
    TiffLayout res;

    uint32_t width = 0, height = 0;
    TIFFGetField( tiff, TIFFTAG_IMAGEWIDTH, &width );
    TIFFGetField( tiff, TIFFTAG_IMAGELENGTH, &height );
    constexpr auto maxSize = uint32_t( std::numeric_limits<int>::max() );
    if ( width == 0 || height == 0 || width > maxSize || height > maxSize )
        return unexpected( "Unsupported image size" );
    res.size = Vector2i( int( width ), int( height ) );

    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    uint16_t compression = COMPRESSION_NONE;
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLESPERPIXEL, &res.samplesPerPixel );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_BITSPERSAMPLE, &res.bitsPerSample );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_PLANARCONFIG, &res.planarConfig );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_ORIENTATION, &res.orientation );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_COMPRESSION, &compression );
    if ( !TIFFGetField( tiff, TIFFTAG_PHOTOMETRIC, &res.photometric ) )
        res.photometric = res.samplesPerPixel >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

    // only libtiff's RGBA reader checks the codec, other reads just fail without it
    if ( !TIFFIsCODECConfigured( compression ) )
    {
        const auto* codec = TIFFFindCODEC( compression );
        return unexpected( codec ? fmt::format( "Unsupported compression: {}", codec->name ) : fmt::format( "Unsupported compression: {}", compression ) );
    }

    if ( TIFFIsTiled( tiff ) )
    {
        uint32_t tileWidth = 0, tileHeight = 0, tileDepth = 1;
        TIFFGetField( tiff, TIFFTAG_TILEWIDTH, &tileWidth );
        TIFFGetField( tiff, TIFFTAG_TILELENGTH, &tileHeight );
        TIFFGetFieldDefaulted( tiff, TIFFTAG_TILEDEPTH, &tileDepth );
        if ( tileWidth == 0 || tileHeight == 0 || tileWidth > maxSize || tileHeight > maxSize || tileDepth > 1 )
            return unexpected( "Unsupported tiles format" );
        res.tileSize = Vector2i( int( tileWidth ), int( tileHeight ) );
    }

    // the stored samples, the values and a tile must fit in memory; the largest value takes 8 bytes
    const auto maxPixelSize = std::max( uint64_t( res.samplesPerPixel ) * ( ( res.bitsPerSample + 7u ) / 8u ), uint64_t( 8 ) );
    const auto maxPixels = uint64_t( std::numeric_limits<std::ptrdiff_t>::max() ) / maxPixelSize;
    if ( uint64_t( width ) * height > maxPixels )
        return unexpected( "Image is too large" );
    if ( res.tileSize && uint64_t( res.tileSize->x ) * uint64_t( res.tileSize->y ) > maxPixels )
        return unexpected( "Unsupported tiles format" );

    // the samples are read as stored if they have plain meaning: gray values with one sample per pixel, or RGB colors;
    // the other pixels are decoded by libtiff's RGBA reader, as the old image loader did
    const auto sampleType = getSampleType( sampleFormat, res.bitsPerSample );
    const bool isGray = res.photometric == PHOTOMETRIC_MINISWHITE || res.photometric == PHOTOMETRIC_MINISBLACK;
    char emsg[1024] = {};
    if ( sampleType != ScalarType::Unknown && isGray && res.samplesPerPixel == 1 )
    {
        res.sampleType = res.valueType = sampleType;
    }
    else if ( sampleType != ScalarType::Unknown && res.photometric == PHOTOMETRIC_RGB && res.samplesPerPixel >= 3 )
    {
        // as in libtiff's RGBA reader, the fourth sample is alpha whatever ExtraSamples says
        res.sampleType = sampleType;
        res.valueType = res.samplesPerPixel == 3 ? ScalarType::RGB8 : ScalarType::RGBA8;
    }
    else if ( !TIFFRGBAImageOK( tiff, emsg ) )
    {
        // the formats that libtiff cannot decode keep the stored values of the first sample, as the old image loader read them;
        // except for YCbCr, whose subsampled chroma is not stored by pixels
        if ( sampleType == ScalarType::Unknown || res.photometric == PHOTOMETRIC_YCBCR )
            return unexpected( fmt::format( "Unsupported pixel format: {}", emsg ) );
        res.sampleType = res.valueType = sampleType;
    }
    return res;
}

// reads the samples as stored in the file to dst, in which the pixels of a row and the samples of a pixel go together
Expected<void> readSamples( TIFF* tiff, const TiffLayout& layout, uint8_t* dst, const ProgressCallback& progress )
{
    const auto sampleSize = getScalarTypeSize( layout.sampleType );
    const auto samplesPerPixel = size_t( layout.samplesPerPixel );
    const auto pixelSize = sampleSize * samplesPerPixel;
    const auto width = size_t( layout.size.x );
    const auto height = size_t( layout.size.y );
    const auto rowSize = width * pixelSize;
    // each sample of a pixel is stored in its own plane
    const bool separatePlanes = layout.planarConfig == PLANARCONFIG_SEPARATE && samplesPerPixel > 1;
    const size_t planeCount = separatePlanes ? samplesPerPixel : 1;
    const size_t planePixelSize = separatePlanes ? sampleSize : pixelSize;

    // the file stores the pixels by chunks: tiles, or strips of whole rows
    const bool tiled = layout.tileSize.has_value();
    size_t chunkWidth = width;
    size_t chunkHeight = 0;
    if ( tiled )
    {
        chunkWidth = size_t( layout.tileSize->x );
        chunkHeight = size_t( layout.tileSize->y );
    }
    else
    {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted( tiff, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip );
        chunkHeight = std::clamp( size_t( rowsPerStrip ), size_t( 1 ), height );
    }
    const auto chunkRowSize = chunkWidth * planePixelSize;
    // strips of contiguous samples are read directly to dst, other chunks through this buffer
    const bool direct = !tiled && !separatePlanes;
    Buffer<uint8_t> buffer( direct ? 0 : chunkRowSize * chunkHeight );

    const auto chunkTotal = planeCount * chunkCount( height, chunkHeight ) * chunkCount( width, chunkWidth );
    size_t chunksRead = 0;
    for ( size_t plane = 0; plane < planeCount; ++plane )
    {
        for ( const auto chunkY : splitByChunks( height, chunkHeight ) )
        {
            for ( const auto chunkX : splitByChunks( width, chunkWidth ) )
            {
                auto* chunkData = direct ? dst + chunkY.offset * rowSize : buffer.data();
                if ( tiled )
                {
                    const auto tile = TIFFComputeTile( tiff, uint32_t( chunkX.offset ), uint32_t( chunkY.offset ), 0, uint16_t( plane ) );
                    if ( TIFFReadEncodedTile( tiff, tile, chunkData, tmsize_t( buffer.size() ) ) < 0 )
                        return unexpected( "Error reading tile" );
                }
                else
                {
                    const auto strip = TIFFComputeStrip( tiff, uint32_t( chunkY.offset ), uint16_t( plane ) );
                    if ( TIFFReadEncodedStrip( tiff, strip, chunkData, tmsize_t( chunkY.size * chunkRowSize ) ) < 0 )
                        return unexpected( "Error reading strip" );
                }
                if ( !direct )
                {
                    // the tiles of the last column and of the last row can extend beyond the image
                    for ( size_t row = 0; row < chunkY.size; ++row )
                    {
                        const auto* src = chunkData + row * chunkRowSize;
                        auto* d = dst + ( chunkY.offset + row ) * rowSize + chunkX.offset * pixelSize;
                        if ( separatePlanes )
                        {
                            for ( size_t x = 0; x < chunkX.size; ++x )
                                std::memcpy( d + x * pixelSize + plane * sampleSize, src + x * sampleSize, sampleSize );
                        }
                        else
                        {
                            std::memcpy( d, src, chunkX.size * pixelSize );
                        }
                    }
                }
                if ( !reportProgress( progress, float( ++chunksRead ) / float( chunkTotal ) ) )
                    return unexpectedOperationCanceled();
            }
        }
    }
    return {};
}

// converts a color or alpha sample to 8 bits: as in libtiff's RGBA reader, integer samples are taken as unsigned ones,
// and 16-bit samples are rounded; wider integer samples keep their high byte; floating-point samples are clamped to [0, 1], NaN becomes 0
template <typename T>
uint8_t colorTo8Bit( T v )
{
    if constexpr ( std::is_floating_point_v<T> )
    {
        return std::isnan( v ) ? uint8_t( 0 ) : Color::valToUint8( v );
    }
    else
    {
        const auto u = std::make_unsigned_t<T>( v );
        if constexpr ( sizeof( T ) == 1 )
            return u;
        else if constexpr ( sizeof( T ) == 2 )
            return uint8_t( ( u + 128u ) / 257u );
        else
            return uint8_t( u >> ( 8 * sizeof( T ) - 8 ) );
    }
}

// converts the samples as stored in the file to the values of the raster: takes the first sample of each pixel,
// or converts the first three or four samples to 8-bit color components
void convertSamples( const TiffLayout& layout, const uint8_t* src, uint8_t* dst )
{
    const auto pixelCount = size_t( layout.size.x ) * size_t( layout.size.y );
    const auto sampleSize = getScalarTypeSize( layout.sampleType );
    const auto storedPixelSize = size_t( layout.samplesPerPixel ) * sampleSize;
    if ( layout.valueType == layout.sampleType )
    {
        parallelForEach( pixelCount, [&] ( size_t i )
        {
            std::memcpy( dst + i * sampleSize, src + i * storedPixelSize, sampleSize );
        } );
        return;
    }

    const auto channels = getScalarTypeSize( layout.valueType );
    parallelForEach( pixelCount, [&] ( size_t i )
    {
        for ( size_t c = 0; c < channels; ++c )
        {
            const auto* sample = (const char*)src + i * storedPixelSize + c * sampleSize;
            dst[i * channels + c] = visitScalarType( [] ( auto v ) { return colorTo8Bit( v ); }, layout.sampleType, sample );
        }
    } );
}

// reorders the pixels from the stored order to the one with the top-left pixel first,
// the orientations with swapped rows and columns are treated as the ones without the swap, as libtiff's RGBA reader does
void applyOrientation( uint8_t* data, const Vector2i& size, size_t pixelSize, uint16_t orientation )
{
    bool flipX = false;
    bool flipY = false;
    switch ( orientation )
    {
    case ORIENTATION_TOPRIGHT:
    case ORIENTATION_RIGHTTOP:
        flipX = true;
        break;
    case ORIENTATION_BOTRIGHT:
    case ORIENTATION_RIGHTBOT:
        flipX = true;
        flipY = true;
        break;
    case ORIENTATION_BOTLEFT:
    case ORIENTATION_LEFTBOT:
        flipY = true;
        break;
    default:
        break;
    }

    const auto width = size_t( size.x );
    const auto height = size_t( size.y );
    const auto rowSize = width * pixelSize;
    if ( flipY )
    {
        ParallelFor( size_t( 0 ), height / 2, [&] ( size_t y )
        {
            auto* row = data + y * rowSize;
            std::swap_ranges( row, row + rowSize, data + ( height - 1 - y ) * rowSize );
        } );
    }
    if ( flipX )
    {
        ParallelFor( size_t( 0 ), height, [&] ( size_t y )
        {
            auto* row = data + y * rowSize;
            for ( size_t x = 0; x < width / 2; ++x )
                std::swap_ranges( row + x * pixelSize, row + ( x + 1 ) * pixelSize, row + ( width - 1 - x ) * pixelSize );
        } );
    }
}

Expected<Raster> readRaster( TIFF* tiff, const TiffLayout& layout, const ProgressCallback& progress )
{
    Raster res{ .info = layout.rasterInfo() };
    res.data.resize( res.info.dataSize() );

    if ( layout.sampleType == ScalarType::Unknown )
    {
        // TIFFReadRGBAImageOriented stores the pixels as uint32 values with the red component in the lowest byte,
        // which is the order of the components in a little-endian machine
        static_assert( std::endian::native == std::endian::little );
        if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( layout.size.x ), uint32_t( layout.size.y ), (uint32_t*)res.data.data(), ORIENTATION_TOPLEFT, 1 ) )
            return unexpected( "Error reading pixels" );
        if ( !reportProgress( progress, 1.f ) )
            return unexpectedOperationCanceled();
        return res;
    }

    // the samples are read directly to the raster if they need no conversion: one sample per pixel, or 8-bit color samples
    const auto valueSize = getScalarTypeSize( layout.valueType );
    const auto storedPixelSize = size_t( layout.samplesPerPixel ) * getScalarTypeSize( layout.sampleType );
    const bool asStored = storedPixelSize == valueSize;
    Buffer<uint8_t> stored( asStored ? 0 : size_t( layout.size.x ) * size_t( layout.size.y ) * storedPixelSize );
    auto* samples = asStored ? res.data.data() : stored.data();
    if ( auto readRes = readSamples( tiff, layout, samples, progress ); !readRes )
        return unexpected( std::move( readRes.error() ) );
    if ( !asStored )
        convertSamples( layout, samples, res.data.data() );

    applyOrientation( res.data.data(), layout.size, valueSize, layout.orientation );
    return res;
}

Expected<Image> readImage( TIFF* tiff, const TiffLayout& layout )
{
    auto res = readRaster( tiff, layout, {} ).and_then( convertRasterToImage );
    // as in libtiff's RGBA reader, 8-bit and 16-bit gray values are shown inverted if the smallest value means white;
    // other gray values are not, since writeRawTiff marked all files including floating-point distance maps this way;
    // the raster keeps the stored gray values, unlike the colors decoded by libtiff
    if ( res && layout.photometric == PHOTOMETRIC_MINISWHITE && layout.bitsPerSample <= 16 && layout.valueType == layout.sampleType )
    {
        auto& pixels = res->pixels;
        parallelForEach( pixels.size(), [&] ( size_t i )
        {
            auto& c = pixels[i];
            c = Color( 255 - c.r, 255 - c.g, 255 - c.b, c.a );
        } );
    }
    return res;
}

// opens the file for reading and returns f( tiff, layout ); appends the file name to the errors except for the cancellation
template <typename F>
auto readTiffFile( const std::filesystem::path& path, F&& f ) -> decltype( f( (TIFF*)nullptr, TiffLayout{} ) )
{
    const auto file = openTiff( path, "r" );
    if ( !file )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    auto res = readLayout( file.get() ).and_then( [&] ( const TiffLayout& layout ) { return f( file.get(), layout ); } );
    if ( !res && res.error() != stringOperationCanceled() )
        return addFileNameInError( std::move( res ), path );
    return res;
}

// GeoTIFF and GDAL tags written with distance maps
struct GeoTags
{
    // optional transformation of (column, row, value) to world coordinates
    const AffineXf3f* pixelToWorld = nullptr;
    // optional value meaning no data
    std::optional<double> noData;
};

// writes a raster with one layer, getRow returns the values of given row, the rows go from top to bottom
Expected<void> writeTiff( const std::filesystem::path& path, const RasterInfo& info, const std::function<const uint8_t* ( size_t )>& getRow,
    const ProgressCallback& progress, const GeoTags& geoTags = {} )
{
    // the color components are stored as 8-bit samples, other values as single samples
    const bool isColor = info.type == ScalarType::RGB8 || info.type == ScalarType::RGBA8;
    const auto sampleType = isColor ? ScalarType::UInt8 : info.type;
    const auto* sample = std::find_if( std::begin( cSampleTypes ), std::end( cSampleTypes ), [sampleType] ( const SampleType& s ) { return s.type == sampleType; } );
    if ( sample == std::end( cSampleTypes ) )
        return unexpected( "Unsupported value type" );
    if ( info.dims.x <= 0 || info.dims.y <= 0 || info.dims.z <= 0 )
        return unexpected( "Cannot save empty raster" );
    if ( info.dims.z != 1 )
        return unexpected( "Cannot save a raster with several layers" );

    const auto file = openTiff( path, "w" );
    if ( !file )
        return unexpected( "Cannot write file: " + utf8string( path ) );
    TIFF* tiff = file.get();

    TIFFSetField( tiff, TIFFTAG_IMAGEWIDTH, uint32_t( info.dims.x ) );
    TIFFSetField( tiff, TIFFTAG_IMAGELENGTH, uint32_t( info.dims.y ) );
    TIFFSetField( tiff, TIFFTAG_BITSPERSAMPLE, sample->bits );
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, isColor ? int( getScalarTypeSize( info.type ) ) : 1 );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sample->format );
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, isColor ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );
    if ( info.type == ScalarType::RGBA8 )
    {
        const uint16_t extraSample = EXTRASAMPLE_UNASSALPHA;
        TIFFSetField( tiff, TIFFTAG_EXTRASAMPLES, 1, &extraSample );
    }

    // declare non-standard tags
    std::vector<TIFFFieldInfo> fieldInfo;
    if ( geoTags.pixelToWorld )
        fieldInfo.push_back( { cModelTransformationTag, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTransformationTag" } );
    if ( geoTags.noData )
        fieldInfo.push_back( { cGdalNoDataTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GDALNoDataValue" } );
    if ( !fieldInfo.empty() )
        TIFFMergeFieldInfo( tiff, fieldInfo.data(), uint32_t( fieldInfo.size() ) );

    if ( geoTags.pixelToWorld )
    {
        const Matrix4d matrix = AffineXf3d( *geoTags.pixelToWorld );
        TIFFSetField( tiff, cModelTransformationTag, 16, &matrix );
    }
    if ( geoTags.noData )
        TIFFSetField( tiff, cGdalNoDataTag, fmt::format( "{}", *geoTags.noData ).c_str() );

    const auto height = size_t( info.dims.y );
    for ( size_t row = 0; row < height; ++row )
    {
        if ( TIFFWriteScanline( tiff, (void*)getRow( row ), uint32_t( row ), 0 ) < 0 )
            return unexpected( "Error writing file: " + utf8string( path ) );
        if ( !reportProgress( progress, float( row + 1 ) / float( height ) ) )
            return unexpectedOperationCanceled();
    }
    if ( !TIFFFlush( tiff ) )
        return unexpected( "Error writing file: " + utf8string( path ) );

    return {};
}

} // namespace

namespace MR
{

namespace RasterLoad
{

Expected<Raster> fromTiff( const std::filesystem::path& path, const RasterLoadSettings& settings )
{
    MR_TIMER;
    return readTiffFile( path, [&] ( TIFF* tiff, const TiffLayout& layout ) { return readRaster( tiff, layout, settings.progress ); } );
}

Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path )
{
    return readTiffFile( path, [] ( TIFF*, const TiffLayout& layout ) -> Expected<RasterInfo> { return layout.rasterInfo(); } );
}

MR_ADD_RASTER_LOADER( IOFilter( "TIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff, infoFromTiff )

} // namespace RasterLoad

namespace RasterSave
{

Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings )
{
    MR_TIMER;
    if ( raster.data.size() != raster.info.dataSize() )
        return unexpected( "Raster data size does not match its dimensions" );

    const auto rowSize = size_t( raster.info.dims.x ) * getScalarTypeSize( raster.info.type );
    return writeTiff( path, raster.info, [&] ( size_t row ) { return raster.data.data() + row * rowSize; }, settings.progress );
}

MR_ADD_RASTER_SAVER( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff )
MR_ADD_RASTER_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace RasterSave

namespace ImageLoad
{

Expected<Image> fromTiff( const std::filesystem::path& path )
{
    MR_TIMER;
    return readTiffFile( path, readImage );
}

MR_ADD_IMAGE_LOADER_WITH_PRIORITY( IOFilter( "TIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff, -1 )

} // namespace ImageLoad

namespace ImageSave
{

Expected<void> toTiff( const Image& image, const std::filesystem::path& path )
{
    if ( image.pixels.size() != size_t( std::max( image.resolution.x, 0 ) ) * size_t( std::max( image.resolution.y, 0 ) ) )
        return unexpected( "Image size does not match its resolution" );

    const RasterInfo info{
        .dims = Vector3i( image.resolution.x, image.resolution.y, 1 ),
        .type = ScalarType::RGBA8,
    };
    const auto width = size_t( image.resolution.x );
    const auto height = size_t( image.resolution.y );
    // Image starts from the bottom row
    return writeTiff( path, info, [&] ( size_t row ) { return (const uint8_t*)( image.pixels.data() + ( height - 1 - row ) * width ); }, {} );
}

MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff, -1 )
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff, -1 )

} // namespace ImageSave

namespace DistanceMapSave
{

Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings )
{
    const RasterInfo info{
        .dims = Vector3i( dmap.dims().x, dmap.dims().y, 1 ),
        .type = ScalarType::Float32,
    };
    const auto width = dmap.resX();
    return writeTiff( path, info, [&] ( size_t row ) { return (const uint8_t*)( dmap.data() + row * width ); }, settings.progress, {
        .pixelToWorld = settings.xf,
        .noData = DistanceMap::NOT_VALID_VALUE,
    } );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace DistanceMapSave

} // namespace MR
#endif
