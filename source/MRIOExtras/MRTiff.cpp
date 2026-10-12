#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRFmt.h"

#include <tiffio.h>

#include <bit>
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
    // libtiff decodes the pixels to RGBA colors
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
    if ( !( storable && plain ) && TIFFRGBAImageOK( tiff, emsg ) )
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

Expected<Raster> readRaster( TIFF* tiff, const TiffLayout& layout, const ProgressCallback& progress )
{
    Raster res{ .info = layout.info };
    res.data.resize( res.info.dataSize() );
    if ( !layout.decoded )
    {
        if ( auto readRes = readSamples( tiff, layout, res.data.data(), progress ); !readRes )
            return unexpected( std::move( readRes.error() ) );
        return res;
    }

    const auto& dims = res.info.dims;
    if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( dims.x ), uint32_t( dims.y ), (uint32_t*)res.data.data(), ORIENTATION_TOPLEFT, 1 ) )
        return unexpected( "Error reading pixels" );
    if ( !reportProgress( progress, 1.f ) )
        return unexpectedOperationCanceled();
    return res;
}

// as the old image loader did, libtiff decodes the images it can, the others are converted from the raster
Expected<Image> readImage( TIFF* tiff, const TiffLayout& layout )
{
    char emsg[1024] = {};
    if ( !TIFFRGBAImageOK( tiff, emsg ) )
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
    // ExtraSamples is not written, as before, so libtiff's RGBA reader takes the fourth sample as associated alpha
    // and does not premultiply the colors by it
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, info.channels );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sample->format );
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, info.channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB );
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
    const auto rowSize = size_t( info.dims.x ) * size_t( info.channels ) * getScalarTypeSize( info.type );
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
    return readTiffFile( path, [] ( TIFF*, const TiffLayout& layout ) -> Expected<RasterInfo> { return layout.info; } );
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
        .type = ScalarType::UInt8,
        .channels = 4,
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
