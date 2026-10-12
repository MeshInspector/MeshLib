#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRParallelMinMax.h"
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

// GeoTIFF tags: http://geotiff.maptools.org/spec/geotiff2.6.html
constexpr uint32_t cModelPixelScaleTag = 33550;
constexpr uint32_t cModelTiePointTag = 33922;
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

// returns how the values of given type are stored, or nullptr if they cannot be
const SampleType* findSampleType( ScalarType type )
{
    const auto* res = std::find_if( std::begin( cSampleTypes ), std::end( cSampleTypes ), [type] ( const SampleType& s ) { return s.type == type; } );
    return res != std::end( cSampleTypes ) ? res : nullptr;
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

// one channel of Float32 values is read straight to the floats
bool storesFloats( const TiffLayout& layout )
{
    return layout.info.type == ScalarType::Float32 && layout.info.channels == 1;
}

// reads the values of other formats and converts them to floats by convertRasterToFloats
Expected<std::vector<float>> readConvertedFloats( TIFF* tiff, const TiffLayout& layout, const ProgressCallback& progress )
{
    std::vector<float> res;
    return readRaster( tiff, layout, progress )
        .and_then( [&] ( const Raster& raster ) { return convertRasterToFloats( raster, res ); } )
        .transform( [&] { return std::move( res ); } );
}

// returns the values of a tag of doubles and sets their count, or returns nullptr; libtiff passes the count as a 32-bit value
// for the tags it does not know, and as a 16-bit value for the tags registered with TIFF_VARIABLE, as libgeotiff's tag extender does
const double* getDoublesTag( TIFF* tiff, uint32_t tag, uint32_t& count )
{
    count = 0;
    double* data = nullptr;
    // the tags of other types or with a fixed number of values are returned differently
    const auto* field = TIFFFindField( tiff, tag, TIFF_ANY );
    if ( !field || TIFFFieldDataType( field ) != TIFF_DOUBLE || !TIFFFieldPassCount( field ) )
        return nullptr;
    if ( TIFFFieldReadCount( field ) == TIFF_VARIABLE2 )
        return TIFFGetField( tiff, tag, &count, &data ) ? data : nullptr;
    uint16_t count16 = 0;
    if ( !TIFFGetField( tiff, tag, &count16, &data ) )
        return nullptr;
    count = count16;
    return data;
}

// reads the GeoTIFF transformation of (column, row, value) to world coordinates
std::optional<AffineXf3f> readGeoTransform( TIFF* tiff )
{
    uint32_t count = 0;
    // the values are freed with the file
    if ( const auto* matrix = getDoublesTag( tiff, cModelTransformationTag, count ); matrix && count == 16 )
        return AffineXf3f( Matrix4f( *(const Matrix4d*)matrix ) );

    const auto* tie = getDoublesTag( tiff, cModelTiePointTag, count );
    if ( !tie || count != 6 )
        return {};
    const auto* scaleTag = getDoublesTag( tiff, cModelPixelScaleTag, count );
    if ( !scaleTag || count != 3 )
        return {};
    Vector3d tiePoints[2] = { { tie[0], tie[1], tie[2] }, { tie[3], tie[4], tie[5] } };
    Vector3d scale{ scaleTag[0], -scaleTag[1], scaleTag[2] };
    if ( scale.z == 0. )
    {
        tiePoints[1].z = 0.;
        scale.z = 1.;
    }
    return AffineXf3f( Matrix3f::scale( Vector3f( scale ) ), Vector3f( tiePoints[0] + tiePoints[1] ) );
}

// describes the pixels as the old raster reader did; the colors decoded by libtiff are 8-bit RGBA ones
Expected<TiffParameters> getTiffParameters( const TiffLayout& layout )
{
    const auto& info = layout.info;
    TiffParameters res;
    switch ( info.channels )
    {
    case 1: res.valueType = TiffParameters::ValueType::Scalar; break;
    case 3: res.valueType = TiffParameters::ValueType::RGB; break;
    case 4: res.valueType = TiffParameters::ValueType::RGBA; break;
    default: return unexpected( "Unsupported pixel format" );
    }
    // the layout has only the types of stored samples
    const auto* sample = findSampleType( info.type );
    assert( sample );
    res.sampleType = sample->format == SAMPLEFORMAT_IEEEFP ? TiffParameters::SampleType::Float
        : sample->format == SAMPLEFORMAT_INT ? TiffParameters::SampleType::Int : TiffParameters::SampleType::Uint;
    res.bytesPerSample = sample->bits / 8;
    res.imageSize = Vector2i( info.dims.x, info.dims.y );
    res.tiled = layout.tileSize.has_value();
    res.tileSize = layout.tileSize.value_or( Vector2i() );
    return res;
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

// GeoTIFF and GDAL tags written with distance maps
struct GeoTags
{
    // optional transformation of (column, row, value) to world coordinates
    const AffineXf3f* pixelToWorld = nullptr;
    // optional value meaning no data, not written if empty
    std::string noData;
};

// writes a raster with one layer, the rows of data go from top to bottom, or from bottom to top as in Image if bottomUp is set
Expected<void> writeTiff( const std::filesystem::path& path, const RasterInfo& info, const uint8_t* data, bool bottomUp,
    const ProgressCallback& progress, const GeoTags& geoTags = {} )
{
    const auto* sample = findSampleType( info.type );
    if ( !sample )
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
    if ( !geoTags.noData.empty() )
        fieldInfo.push_back( { cGdalNoDataTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GDALNoDataValue" } );
    if ( !fieldInfo.empty() )
        TIFFMergeFieldInfo( tiff, fieldInfo.data(), uint32_t( fieldInfo.size() ) );

    if ( geoTags.pixelToWorld )
    {
        const Matrix4d matrix = AffineXf3d( *geoTags.pixelToWorld );
        TIFFSetField( tiff, cModelTransformationTag, 16, &matrix );
    }
    if ( !geoTags.noData.empty() )
        TIFFSetField( tiff, cGdalNoDataTag, geoTags.noData.c_str() );

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

namespace DistanceMapLoad
{

Expected<DistanceMap> fromTiff( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
{
    MR_TIMER;
    return readTiffFile( path, [&] ( TIFF* tiff, const TiffLayout& layout ) -> Expected<DistanceMap>
    {
        DistanceMap res( layout.info.dims.x, layout.info.dims.y );
        auto readRes = storesFloats( layout ) ? readPixels( tiff, layout, (uint8_t*)res.data(), settings.progress )
            : readConvertedFloats( tiff, layout, settings.progress ).transform( [&] ( std::vector<float>&& values ) { res.set( std::move( values ) ); } );
        if ( !readRes )
            return unexpected( std::move( readRes.error() ) );
        if ( settings.distanceMapToWorld )
            *settings.distanceMapToWorld = readGeoTransform( tiff ).value_or( AffineXf3f() );
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
    return writeTiff( path, info, (const uint8_t*)dmap.data(), false, settings.progress, {
        .pixelToWorld = settings.xf,
        .noData = fmt::format( "{}", double( DistanceMap::NOT_VALID_VALUE ) ),
    } );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace DistanceMapSave

bool isTIFFFile( const std::filesystem::path& path )
{
    // only the header is read
    return bool( openTiff( path, "rh" ) );
}

Expected<TiffParameters> readTiffParameters( const std::filesystem::path& path )
{
    return readTiffFile( path, [] ( TIFF*, const TiffLayout& layout ) { return getTiffParameters( layout ); } );
}

Expected<void> readRawTiff( const std::filesystem::path& path, RawTiffOutput& output )
{
    MR_TIMER;
    return readTiffFile( path, [&] ( TIFF* tiff, const TiffLayout& layout ) -> Expected<void>
    {
        const auto params = getTiffParameters( layout );
        if ( !params )
            return unexpected( params.error() );
        if ( output.params )
            *output.params = *params;
        if ( output.p2wXf )
            if ( const auto xf = readGeoTransform( tiff ) )
                *output.p2wXf = *xf;

        const auto pixels = size_t( layout.info.dims.x ) * size_t( layout.info.dims.y );
        if ( !output.bytes || output.size < ( output.convertToFloat ? pixels * sizeof( float ) : layout.info.dataSize() ) )
            return unexpected( "Output buffer is too small" );
        if ( !output.convertToFloat )
            return readPixels( tiff, layout, output.bytes, {} );

        auto* floats = (float*)output.bytes;
        // the values of other formats are converted and copied to the output
        auto readRes = storesFloats( layout ) ? readPixels( tiff, layout, output.bytes, {} )
            : readConvertedFloats( tiff, layout, {} ).transform( [&] ( const std::vector<float>& values ) { std::copy( values.begin(), values.end(), floats ); } );
        if ( readRes && ( output.min || output.max ) )
        {
            // the given range is extended to include the values
            const auto [min, max] = parallelMinMax( floats, pixels );
            if ( output.min )
                *output.min = std::min( *output.min, min );
            if ( output.max )
                *output.max = std::max( *output.max, max );
        }
        return readRes;
    } );
}

Expected<void> writeRawTiff( const uint8_t* bytes, const std::filesystem::path& path, const WriteRawTiffParams& params )
{
    MR_TIMER;
    using P = BaseTiffParameters;
    const auto& base = params.baseParams;
    const uint16_t sampleFormat = base.sampleType == P::SampleType::Float ? SAMPLEFORMAT_IEEEFP
        : base.sampleType == P::SampleType::Int ? SAMPLEFORMAT_INT : base.sampleType == P::SampleType::Uint ? SAMPLEFORMAT_UINT : 0;
    const RasterInfo info{
        .dims = Vector3i( base.imageSize.x, base.imageSize.y, 1 ),
        .type = getSampleType( sampleFormat, uint16_t( 8 * base.bytesPerSample ) ),
        .channels = base.valueType == P::ValueType::RGB ? 3 : base.valueType == P::ValueType::RGBA ? 4 : 1,
    };
    return writeTiff( path, info, bytes, false, {}, { .pixelToWorld = params.xf, .noData = params.noData } );
}

} // namespace MR
#endif
