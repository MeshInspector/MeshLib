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

// libtiff's RGBA reader stores the pixels as uint32 values with the red component in the lowest byte,
// which is the order of the components in Color and in RGBA8 values in a little-endian machine
static_assert( std::endian::native == std::endian::little );

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
    std::optional<Vector2i> tileSize;
    // type of the samples if they are read as stored; ScalarType::Unknown if libtiff decodes the pixels to RGBA8 values
    ScalarType sampleType = ScalarType::Unknown;
    // type of the raster values: sampleType for one sample per pixel, otherwise RGB8 or RGBA8 made of the first three or four samples,
    // or decoded by libtiff
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

    uint16_t bitsPerSample = 1, sampleFormat = SAMPLEFORMAT_UINT, planarConfig = PLANARCONFIG_CONTIG, photometric = PHOTOMETRIC_MINISBLACK;
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLESPERPIXEL, &res.samplesPerPixel );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_PLANARCONFIG, &planarConfig );
    if ( !TIFFGetField( tiff, TIFFTAG_PHOTOMETRIC, &photometric ) )
        photometric = res.samplesPerPixel >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

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
    const auto maxPixelSize = std::max( uint64_t( res.samplesPerPixel ) * ( ( bitsPerSample + 7u ) / 8u ), uint64_t( 8 ) );
    const auto maxPixels = uint64_t( std::numeric_limits<std::ptrdiff_t>::max() ) / maxPixelSize;
    if ( uint64_t( width ) * height > maxPixels || ( res.tileSize && uint64_t( res.tileSize->x ) * uint64_t( res.tileSize->y ) > maxPixels ) )
        return unexpected( "Image is too large" );

    // gray values and RGB colors are read as stored, as the old raster reader did; libtiff decodes the other formats, as the old
    // image loader did; if it cannot, they are read as stored too: as gray values or as colors depending on the number of samples
    const auto sampleType = getSampleType( sampleFormat, bitsPerSample );
    const bool storable = sampleType != ScalarType::Unknown && photometric != PHOTOMETRIC_YCBCR
        && ( res.samplesPerPixel == 1 || ( res.samplesPerPixel >= 3 && planarConfig == PLANARCONFIG_CONTIG ) );
    const bool plain = photometric == PHOTOMETRIC_RGB
        || ( res.samplesPerPixel == 1 && ( photometric == PHOTOMETRIC_MINISBLACK || photometric == PHOTOMETRIC_MINISWHITE ) );
    char emsg[1024] = {};
    if ( !( storable && plain ) && TIFFRGBAImageOK( tiff, emsg ) )
        return res;
    if ( !storable )
        return unexpected( fmt::format( "Unsupported pixel format: {}", emsg ) );

    res.sampleType = sampleType;
    if ( res.samplesPerPixel == 1 )
        res.valueType = sampleType;
    else if ( res.samplesPerPixel == 3 )
        res.valueType = ScalarType::RGB8;
    // otherwise RGBA8: as in libtiff's RGBA reader, the fourth sample is alpha whatever ExtraSamples says
    return res;
}

// reads the samples as stored in the file to dst, row by row
Expected<void> readSamples( TIFF* tiff, const TiffLayout& layout, uint8_t* dst, const ProgressCallback& progress )
{
    const auto width = size_t( layout.size.x );
    const auto height = size_t( layout.size.y );
    const auto pixelSize = size_t( layout.samplesPerPixel ) * getScalarTypeSize( layout.sampleType );
    const auto rowSize = width * pixelSize;
    if ( !layout.tileSize )
    {
        for ( size_t y = 0; y < height; ++y )
        {
            if ( TIFFReadScanline( tiff, dst + y * rowSize, uint32_t( y ), 0 ) < 0 )
                return unexpected( "Error reading row" );
            if ( !reportProgress( progress, float( y + 1 ) / float( height ) ) )
                return unexpectedOperationCanceled();
        }
        return {};
    }

    // the tiles of the last column and of the last row can extend beyond the image
    const auto tileWidth = size_t( layout.tileSize->x );
    const auto tileHeight = size_t( layout.tileSize->y );
    const auto tileRowSize = tileWidth * pixelSize;
    Buffer<uint8_t> buffer( tileRowSize * tileHeight );
    const auto tileCount = chunkCount( width, tileWidth ) * chunkCount( height, tileHeight );
    size_t tilesRead = 0;
    for ( const auto tileY : splitByChunks( height, tileHeight ) )
    {
        for ( const auto tileX : splitByChunks( width, tileWidth ) )
        {
            if ( TIFFReadTile( tiff, buffer.data(), uint32_t( tileX.offset ), uint32_t( tileY.offset ), 0, 0 ) < 0 )
                return unexpected( "Error reading tile" );
            for ( size_t row = 0; row < tileY.size; ++row )
                std::memcpy( dst + ( tileY.offset + row ) * rowSize + tileX.offset * pixelSize, buffer.data() + row * tileRowSize, tileX.size * pixelSize );
            if ( !reportProgress( progress, float( ++tilesRead ) / float( tileCount ) ) )
                return unexpectedOperationCanceled();
        }
    }
    return {};
}

// converts a color sample to 8 bits: integer samples keep their high byte and are taken as unsigned ones, as in libtiff's RGBA reader;
// floating-point samples are clamped to [0, 1], NaN becomes 0
template <typename T>
uint8_t colorTo8Bit( T v )
{
    if constexpr ( std::is_floating_point_v<T> )
        return std::isnan( v ) ? uint8_t( 0 ) : Color::valToUint8( v );
    else
        return uint8_t( std::make_unsigned_t<T>( v ) >> ( 8 * sizeof( T ) - 8 ) );
}

// converts the first three or four samples of each pixel as stored in the file to 8-bit color components
void convertColors( const TiffLayout& layout, const uint8_t* src, uint8_t* dst )
{
    const auto sampleSize = getScalarTypeSize( layout.sampleType );
    const auto storedPixelSize = size_t( layout.samplesPerPixel ) * sampleSize;
    const auto channels = getScalarTypeSize( layout.valueType );
    ParallelFor( size_t( 0 ), size_t( layout.size.x ) * size_t( layout.size.y ), [&] ( size_t i )
    {
        for ( size_t c = 0; c < channels; ++c )
        {
            const auto* sample = (const char*)src + i * storedPixelSize + c * sampleSize;
            dst[i * channels + c] = visitScalarType( [] ( auto v ) { return colorTo8Bit( v ); }, layout.sampleType, sample );
        }
    } );
}

Expected<Raster> readRaster( TIFF* tiff, const TiffLayout& layout, const ProgressCallback& progress )
{
    Raster res{ .info = layout.rasterInfo() };
    res.data.resize( res.info.dataSize() );

    if ( layout.sampleType == ScalarType::Unknown )
    {
        if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( layout.size.x ), uint32_t( layout.size.y ), (uint32_t*)res.data.data(), ORIENTATION_TOPLEFT, 1 ) )
            return unexpected( "Error reading pixels" );
        if ( !reportProgress( progress, 1.f ) )
            return unexpectedOperationCanceled();
        return res;
    }

    // the samples are read directly to the raster if they need no conversion: one sample per pixel, or three or four 8-bit color samples
    const auto storedPixelSize = size_t( layout.samplesPerPixel ) * getScalarTypeSize( layout.sampleType );
    const bool asStored = storedPixelSize == getScalarTypeSize( layout.valueType );
    Buffer<uint8_t> stored( asStored ? 0 : size_t( layout.size.x ) * size_t( layout.size.y ) * storedPixelSize );
    auto* samples = asStored ? res.data.data() : stored.data();
    if ( auto readRes = readSamples( tiff, layout, samples, progress ); !readRes )
        return unexpected( std::move( readRes.error() ) );
    if ( !asStored )
        convertColors( layout, samples, res.data.data() );
    return res;
}

// as the old image loader did, libtiff decodes the images it can, the others are converted from the raster
Expected<Image> readImage( TIFF* tiff, const TiffLayout& layout )
{
    char emsg[1024] = {};
    if ( !TIFFRGBAImageOK( tiff, emsg ) )
        return readRaster( tiff, layout, {} ).and_then( convertRasterToImage );

    Image res{ .resolution = layout.size };
    res.pixels.resize( size_t( layout.size.x ) * size_t( layout.size.y ) );
    // ORIENTATION_BOTLEFT puts the bottom row first, as in Image
    if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( layout.size.x ), uint32_t( layout.size.y ), (uint32_t*)res.pixels.data(), ORIENTATION_BOTLEFT, 1 ) )
        return unexpected( "Error reading pixels" );
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

// writes a raster with one layer, the rows of data go from top to bottom, or from bottom to top as in Image if bottomUp is set
Expected<void> writeTiff( const std::filesystem::path& path, const RasterInfo& info, const uint8_t* data, bool bottomUp,
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
    // ExtraSamples is not written, as before, so libtiff's RGBA reader takes the fourth sample as associated alpha
    // and does not premultiply the colors by it
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, isColor ? int( getScalarTypeSize( info.type ) ) : 1 );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sample->format );
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, isColor ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );

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
    const auto rowSize = size_t( info.dims.x ) * getScalarTypeSize( info.type );
    for ( size_t row = 0; row < height; ++row )
    {
        const auto* rowData = data + ( bottomUp ? height - 1 - row : row ) * rowSize;
        if ( TIFFWriteScanline( tiff, (void*)rowData, uint32_t( row ), 0 ) < 0 )
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
    return writeTiff( path, raster.info, raster.data.data(), false, settings.progress );
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
    MR_TIMER;
    if ( image.pixels.size() != size_t( image.resolution.x ) * size_t( image.resolution.y ) )
        return unexpected( "Image size does not match its resolution" );
    const RasterInfo info{
        .dims = Vector3i( image.resolution.x, image.resolution.y, 1 ),
        .type = ScalarType::RGBA8,
    };
    return writeTiff( path, info, (const uint8_t*)image.pixels.data(), true, {} );
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
    return writeTiff( path, info, (const uint8_t*)dmap.data(), false, settings.progress, {
        .pixelToWorld = settings.xf,
        .noData = DistanceMap::NOT_VALID_VALUE,
    } );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace DistanceMapSave

} // namespace MR
#endif
