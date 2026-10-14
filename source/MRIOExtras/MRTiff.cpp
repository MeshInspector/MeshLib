#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRSpdlog.h"

#include <tiffio.h>

#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>

namespace
{
using namespace MR;

// GeoTIFF tags: http://geotiff.maptools.org/spec/geotiff2.6.html
constexpr uint32_t cModelPixelScaleTag = 33550;
constexpr uint32_t cModelTiePointTag = 33922;
constexpr uint32_t cModelTransformationTag = 34264;
constexpr uint32_t cGeoKeyDirectoryTag = 34735;
// the raster type key and its value for the rasters whose pixels are points: https://docs.ogc.org/is/19-008r4/19-008r4.html#_raster_space
constexpr uint16_t cGTRasterTypeGeoKey = 1025;
constexpr uint16_t cRasterPixelIsPoint = 2;
// no-data value: https://gdal.org/en/stable/drivers/raster/gtiff.html#nodata-value
constexpr uint32_t cGdalNoDataTag = 42113;

// libtiff's RGBA reader stores the pixels as uint32 values with the red component in the lowest byte,
// which is the order of the components in Color and in the channels of a raster in a little-endian machine
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

// libtiff prints its warnings and errors to stderr; they go to the log instead, as OpenCASCADE's messages do
void logTiffMessage( spdlog::level::level_enum level, const char* module, const char* fmt, va_list ap )
{
    char message[1024];
    std::vsnprintf( message, sizeof( message ), fmt, ap );
    if ( module )
        spdlog::log( level, "libtiff: {}: {}", module, message );
    else
        spdlog::log( level, "libtiff: {}", message );
}

// libtiff before 4.7 warns about every tag it does not know, so the tags of GDAL's GeoTIFF files are registered for each file
// before libtiff reads its directory, as libgeotiff and GDAL define them
const TIFFFieldInfo cGeoTiffFields[] = {
    { cModelPixelScaleTag, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelPixelScaleTag" },
    { cModelTiePointTag, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTiePointTag" },
    { cModelTransformationTag, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTransformationTag" },
    { 34735, -1, -1, TIFF_SHORT, FIELD_CUSTOM, 1, 1, (char*)"GeoKeyDirectoryTag" },
    { 34736, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"GeoDoubleParamsTag" },
    { 34737, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GeoAsciiParamsTag" },
    { 42112, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GDALMetadata" },
    { cGdalNoDataTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GDALNoDataValue" },
};

// the tag extender is process-wide, so it calls the one set before it, as libgeotiff's does
TIFFExtendProc sParentTagExtender = nullptr;
void extendTags( TIFF* tiff )
{
    TIFFMergeFieldInfo( tiff, cGeoTiffFields, uint32_t( std::size( cGeoTiffFields ) ) );
    if ( sParentTagExtender )
        sParentTagExtender( tiff );
}

MR_ON_INIT
{
    TIFFSetWarningHandler( [] ( const char* module, const char* fmt, va_list ap ) { logTiffMessage( spdlog::level::warn, module, fmt, ap ); } );
    TIFFSetErrorHandler( [] ( const char* module, const char* fmt, va_list ap ) { logTiffMessage( spdlog::level::err, module, fmt, ap ); } );
    sParentTagExtender = TIFFSetTagExtender( extendTags );
};

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
    // the stored samples, or four UInt8 channels of the RGBA colors decoded by libtiff
    RasterInfo info;
    // libtiff can decode the pixels to RGBA colors
    bool decodable = false;
    // the raster holds the colors decoded by libtiff
    bool decoded = false;
    std::optional<Vector2i> tileSize;
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
    res.info.dims = Vector3i( int( width ), int( height ), 1 );

    // libtiff opens the files compressed by the methods it does not support and fails only when reading the pixels
    uint16_t compression = COMPRESSION_NONE;
    TIFFGetFieldDefaulted( tiff, TIFFTAG_COMPRESSION, &compression );
    if ( !TIFFIsCODECConfigured( compression ) )
    {
        const auto* codec = TIFFFindCODEC( compression );
        return unexpected( "Unsupported compression method: " + ( codec ? std::string( codec->name ) : std::to_string( compression ) ) );
    }

    uint16_t samplesPerPixel = 1, bitsPerSample = 1, sampleFormat = SAMPLEFORMAT_UINT, planarConfig = PLANARCONFIG_CONTIG,
        photometric = PHOTOMETRIC_MINISBLACK;
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_PLANARCONFIG, &planarConfig );
    if ( !TIFFGetField( tiff, TIFFTAG_PHOTOMETRIC, &photometric ) )
        photometric = samplesPerPixel >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

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

    // the stored samples or the decoded colors must fit in memory, and so must a tile
    const auto maxPixelSize = std::max( uint64_t( samplesPerPixel ) * ( ( bitsPerSample + 7u ) / 8u ), uint64_t( 4 ) );
    const auto maxPixels = uint64_t( std::numeric_limits<std::ptrdiff_t>::max() ) / maxPixelSize;
    if ( uint64_t( width ) * height > maxPixels || ( res.tileSize && uint64_t( res.tileSize->x ) * uint64_t( res.tileSize->y ) > maxPixels ) )
        return unexpected( "Image is too large" );

    // the samples of gray, palette and RGB images are read as stored, as the old raster reader did; libtiff decodes the other formats,
    // as the old image loader did; if it cannot, they are read as stored too
    const auto sampleType = getSampleType( sampleFormat, bitsPerSample );
    const bool storable = sampleType != ScalarType::Unknown && photometric != PHOTOMETRIC_YCBCR
        && ( samplesPerPixel == 1 || ( samplesPerPixel >= 3 && planarConfig == PLANARCONFIG_CONTIG ) );
    const bool plain = photometric == PHOTOMETRIC_MINISBLACK || photometric == PHOTOMETRIC_MINISWHITE
        || photometric == PHOTOMETRIC_RGB || photometric == PHOTOMETRIC_PALETTE;
    char emsg[1024] = {};
    res.decodable = TIFFRGBAImageOK( tiff, emsg ) != 0;
    // libtiff multiplies the decoded colors by unassociated alpha, but not by associated alpha; declared associated in memory,
    // the alpha keeps the colors as stored, as in Image; the values are copied, since libtiff frees the old ones first
    uint16_t extraCount = 0;
    const uint16_t* extra = nullptr;
    if ( TIFFGetField( tiff, TIFFTAG_EXTRASAMPLES, &extraCount, &extra ) && extraCount > 0 && extra[0] == EXTRASAMPLE_UNASSALPHA )
    {
        std::vector<uint16_t> associated( extra, extra + extraCount );
        associated[0] = EXTRASAMPLE_ASSOCALPHA;
        TIFFSetField( tiff, TIFFTAG_EXTRASAMPLES, extraCount, associated.data() );
    }
    if ( res.decodable && !( storable && plain ) )
    {
        res.info.type = ScalarType::UInt8;
        res.info.channels = 4;
        res.decoded = true;
        return res;
    }
    if ( !storable )
        return unexpected( fmt::format( "Unsupported pixel format: {}", emsg ) );

    res.info.type = sampleType;
    res.info.channels = samplesPerPixel;
    return res;
}

// reads the samples as stored in the file to dst, row by row
Expected<void> readSamples( TIFF* tiff, const TiffLayout& layout, uint8_t* dst, const ProgressCallback& progress )
{
    const auto width = size_t( layout.info.dims.x );
    const auto height = size_t( layout.info.dims.y );
    const auto pixelSize = size_t( layout.info.channels ) * getScalarTypeSize( layout.info.type );
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

// reads the values of the raster to dst: the stored samples, or the colors decoded by libtiff
Expected<void> readPixels( TIFF* tiff, const TiffLayout& layout, uint8_t* dst, const ProgressCallback& progress )
{
    if ( !layout.decoded )
        return readSamples( tiff, layout, dst, progress );

    // asked for the orientation of the file, libtiff keeps the stored order of the rows, as in the stored samples
    uint16_t orientation = ORIENTATION_TOPLEFT;
    TIFFGetFieldDefaulted( tiff, TIFFTAG_ORIENTATION, &orientation );
    const auto& dims = layout.info.dims;
    if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( dims.x ), uint32_t( dims.y ), (uint32_t*)dst, orientation, 1 ) )
        return unexpected( "Error reading pixels" );
    if ( !reportProgress( progress, 1.f ) )
        return unexpectedOperationCanceled();
    return {};
}

Expected<Raster> readRaster( TIFF* tiff, const TiffLayout& layout, const ProgressCallback& progress )
{
    Raster res{ .info = layout.info };
    res.data.resize( res.info.dataSize() );
    if ( auto readRes = readPixels( tiff, layout, res.data.data(), progress ); !readRes )
        return unexpected( std::move( readRes.error() ) );
    return res;
}

// returns the values of a tag of doubles or shorts and sets their count, or returns nullptr; libtiff passes the count as a 32-bit value
// for the tags it does not know, and as a 16-bit value for the tags registered with TIFF_VARIABLE, as libgeotiff's tag extender does
template <typename T>
const T* getTagValues( TIFF* tiff, uint32_t tag, uint32_t& count )
{
    static_assert( std::is_same_v<T, double> || std::is_same_v<T, uint16_t> );
    count = 0;
    T* data = nullptr;
    // the tags of other types or with a fixed number of values are returned differently
    const auto* field = TIFFFindField( tiff, tag, TIFF_ANY );
    if ( !field || TIFFFieldDataType( field ) != ( std::is_same_v<T, double> ? TIFF_DOUBLE : TIFF_SHORT ) || !TIFFFieldPassCount( field ) )
        return nullptr;
    if ( TIFFFieldReadCount( field ) == TIFF_VARIABLE2 )
        return TIFFGetField( tiff, tag, &count, &data ) ? data : nullptr;
    uint16_t count16 = 0;
    if ( !TIFFGetField( tiff, tag, &count16, &data ) )
        return nullptr;
    count = count16;
    return data;
}

// returns whether the GeoTIFF raster type is PixelIsPoint
bool isPixelIsPoint( TIFF* tiff )
{
    uint32_t count = 0;
    // the header { version 1, revision, minor revision, number of keys } is followed by the keys { id, location, count, value },
    // the location 0 means that the value is in the key
    const auto* keys = getTagValues<uint16_t>( tiff, cGeoKeyDirectoryTag, count );
    if ( !keys || count < 4 || keys[0] != 1 )
        return false;
    const auto end = std::min( count, 4 + 4 * uint32_t( keys[3] ) );
    for ( uint32_t i = 4; i + 4 <= end; i += 4 )
        if ( keys[i] == cGTRasterTypeGeoKey )
            return keys[i + 1] == 0 && keys[i + 3] == cRasterPixelIsPoint;
    return false;
}

// reads the GeoTIFF transformation of (column, row, value) to world coordinates, or returns the identity if the file has none
AffineXf3f readGeoTransform( TIFF* tiff )
{
    AffineXf3d xf;
    uint32_t count = 0;
    // the values are freed with the file
    if ( const auto* matrix = getTagValues<double>( tiff, cModelTransformationTag, count ); matrix && count == 16 )
    {
        xf = AffineXf3d( *(const Matrix4d*)matrix );
    }
    else
    {
        const auto* tie = getTagValues<double>( tiff, cModelTiePointTag, count );
        if ( !tie || count != 6 )
            return {};
        const auto* scaleTag = getTagValues<double>( tiff, cModelPixelScaleTag, count );
        if ( !scaleTag || count != 3 )
            return {};
        // the tie point maps the raster point tiePoints[0] to the model point tiePoints[1]
        Vector3d tiePoints[2] = { { tie[0], tie[1], tie[2] }, { tie[3], tie[4], tie[5] } };
        Vector3d scale{ scaleTag[0], -scaleTag[1], scaleTag[2] };
        if ( scale.z == 0. )
        {
            tiePoints[0].z = 0.;
            tiePoints[1].z = 0.;
            scale.z = 1.;
        }
        xf = AffineXf3d( Matrix3d::scale( scale ), tiePoints[1] - mult( scale, tiePoints[0] ) );
    }
    // the raster point (0, 0) is the center of the first pixel in PixelIsPoint rasters and its corner in PixelIsArea rasters;
    // the returned transformation refers to the corners, as GDAL's does
    if ( isPixelIsPoint( tiff ) )
        xf.b -= xf.A * Vector3d( 0.5, 0.5, 0. );
    return AffineXf3f( xf );
}

// as the old image loader did, libtiff decodes the images it can, the others are converted from the raster
Expected<Image> readImage( TIFF* tiff, const TiffLayout& layout )
{
    if ( !layout.decodable )
        return readRaster( tiff, layout, {} ).and_then( convertRasterToImage );

    const auto& dims = layout.info.dims;
    Image res{ .resolution = Vector2i( dims.x, dims.y ) };
    res.pixels.resize( size_t( dims.x ) * size_t( dims.y ) );
    // ORIENTATION_BOTLEFT puts the bottom row first, as in Image
    if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( dims.x ), uint32_t( dims.y ), (uint32_t*)res.pixels.data(), ORIENTATION_BOTLEFT, 1 ) )
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

// writes a raster with one layer, the rows of data go from top to bottom, or from bottom to top as in Image if bottomUp is set
Expected<void> writeTiff( const std::filesystem::path& path, const RasterInfo& info, const uint8_t* data, bool bottomUp,
    const RasterSaveSettings& settings, std::optional<double> noData = {} )
{
    const auto* sample = std::find_if( std::begin( cSampleTypes ), std::end( cSampleTypes ), [&info] ( const SampleType& s ) { return s.type == info.type; } );
    if ( sample == std::end( cSampleTypes ) )
        return unexpected( "Unsupported value type" );
    if ( info.channels != 1 && info.channels != 3 && info.channels != 4 )
        return unexpected( "Unsupported number of channels" );
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
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, info.channels );
    // the fourth channel is unassociated alpha, as in Image: the colors are not multiplied by it
    const uint16_t unassociatedAlpha = EXTRASAMPLE_UNASSALPHA;
    if ( info.channels == 4 )
        TIFFSetField( tiff, TIFFTAG_EXTRASAMPLES, uint16_t( 1 ), &unassociatedAlpha );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sample->format );
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, info.channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );

    // the GeoTIFF and GDAL tags are registered by extendTags
    if ( settings.xf )
    {
        const Matrix4d matrix = AffineXf3d( *settings.xf );
        TIFFSetField( tiff, cModelTransformationTag, 16, &matrix );
    }
    if ( noData )
        TIFFSetField( tiff, cGdalNoDataTag, fmt::format( "{}", *noData ).c_str() );

    const auto height = size_t( info.dims.y );
    const auto rowSize = size_t( info.dims.x ) * size_t( info.channels ) * getScalarTypeSize( info.type );
    for ( size_t row = 0; row < height; ++row )
    {
        const auto* rowData = data + ( bottomUp ? height - 1 - row : row ) * rowSize;
        if ( TIFFWriteScanline( tiff, (void*)rowData, uint32_t( row ), 0 ) < 0 )
            return unexpected( "Error writing file: " + utf8string( path ) );
        if ( !reportProgress( settings.progress, float( row + 1 ) / float( height ) ) )
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
    return readTiffFile( path, [&] ( TIFF* tiff, const TiffLayout& layout )
    {
        auto res = readRaster( tiff, layout, settings.progress );
        if ( res && settings.xf )
            *settings.xf = readGeoTransform( tiff );
        return res;
    } );
}

Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path )
{
    return readTiffFile( path, [] ( TIFF*, const TiffLayout& layout ) -> Expected<RasterInfo> { return layout.info; } );
}

MR_ADD_RASTER_LOADER( IOFilter( "TIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff, infoFromTiff )

} // namespace RasterLoad

namespace RasterSave
{

Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const TiffSaveSettings& settings )
{
    MR_TIMER;
    if ( raster.data.size() != raster.info.dataSize() )
        return unexpected( "Raster data size does not match its dimensions" );
    return writeTiff( path, raster.info, raster.data.data(), false, settings, settings.noData );
}

Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings )
{
    return toTiff( raster, path, TiffSaveSettings{ settings } );
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
        .type = ScalarType::UInt8,
        .channels = 4,
    };
    return writeTiff( path, info, (const uint8_t*)image.pixels.data(), true, {} );
}

MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff, -1 )
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff, -1 )

} // namespace ImageSave

namespace DistanceMapLoad
{

Expected<DistanceMap> fromTiff( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
{
    MR_TIMER;
    return readTiffFile( path, [&] ( TIFF* tiff, const TiffLayout& layout ) -> Expected<DistanceMap>
    {
        DistanceMap res( layout.info.dims.x, layout.info.dims.y );
        // one channel of Float32 values is read straight to the map, the values of other formats are converted to it
        auto readRes = layout.info.type == ScalarType::Float32 && layout.info.channels == 1
            ? readPixels( tiff, layout, (uint8_t*)res.data(), settings.progress )
            : readRaster( tiff, layout, settings.progress )
                .and_then( [&] ( const Raster& raster ) { return convertRasterToFloats( raster, res.data(), res.size() ); } );
        if ( !readRes )
            return unexpected( std::move( readRes.error() ) );
        if ( settings.distanceMapToWorld )
            *settings.distanceMapToWorld = readGeoTransform( tiff );
        return res;
    } );
}

MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "GeoTIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff )

} // namespace DistanceMapLoad

namespace DistanceMapSave
{

Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings )
{
    MR_TIMER;
    const RasterInfo info{
        .dims = Vector3i( dmap.dims().x, dmap.dims().y, 1 ),
        .type = ScalarType::Float32,
    };
    return writeTiff( path, info, (const uint8_t*)dmap.data(), false, { .xf = settings.xf, .progress = settings.progress },
        DistanceMap::NOT_VALID_VALUE );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "GeoTIFF (.tif)", "*.tif" ), toTiff )
MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "GeoTIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace DistanceMapSave

} // namespace MR
#endif
