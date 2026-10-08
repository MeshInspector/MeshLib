#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRFmt.h"

#include <tiffio.h>

#include <bit>
#include <cstring>
#include <locale>
#include <span>
#include <sstream>

namespace
{
using namespace MR;

// GeoTIFF tags: http://geotiff.maptools.org/spec/geotiff2.6.html
constexpr uint32_t cModelPixelScaleTag = 33550;
constexpr uint32_t cModelTiepointTag = 33922;
constexpr uint32_t cModelTransformationTag = 34264;
// no-data value: https://gdal.org/en/stable/drivers/raster/gtiff.html#nodata-value
constexpr uint32_t cGdalNoDataTag = 42113;

// on Windows, narrow TIFFOpen interprets the file name in the ANSI code page, which fails for non-ASCII names
TIFF* openTiff( const std::filesystem::path& path, const char* mode )
{
#ifdef _WIN32
    return TIFFOpenW( path.wstring().c_str(), mode );
#else
    return TIFFOpen( utf8string( path ).c_str(), mode );
#endif
}

class TiffHolder
{
public:
    TiffHolder( const std::filesystem::path& path, const char* mode )
    {
        tiffPtr_ = openTiff( path, mode );
    }
    ~TiffHolder()
    {
        if ( !tiffPtr_ )
            return;
        TIFFClose( tiffPtr_ );
        tiffPtr_ = nullptr;
    }
    operator TIFF* ( ) { return tiffPtr_; }
    operator const TIFF* ( ) const { return tiffPtr_; }
    operator bool() const { return bool( tiffPtr_ ); }

private:
    TIFF* tiffPtr_{ nullptr };
};

// appends the file name to the errors except for the cancellation
template <typename T>
Expected<T> addFileName( Expected<T> res, const std::filesystem::path& path )
{
    if ( !res && res.error() != stringOperationCanceled() )
        return addFileNameInError( std::move( res ), path );
    return res;
}

// how the pixels are stored in the file
struct TiffLayout
{
    Vector2i size;
    uint16_t samplesPerPixel = 1;
    uint16_t bitsPerSample = 1;
    uint16_t photometric = PHOTOMETRIC_MINISBLACK;
    uint16_t planarConfig = PLANARCONFIG_CONTIG;
    uint16_t orientation = ORIENTATION_TOPLEFT;
    std::optional<Vector2i> tileSize;
    // type of the samples if they are read as stored; otherwise libtiff decodes the pixels to 8-bit RGBA
    std::optional<ScalarType> sampleType;
};

std::optional<ScalarType> getSampleType( uint16_t sampleFormat, uint16_t bitsPerSample )
{
    switch ( sampleFormat )
    {
    case SAMPLEFORMAT_UINT:
    case SAMPLEFORMAT_VOID:
        switch ( bitsPerSample )
        {
        case 8:
            return ScalarType::UInt8;
        case 16:
            return ScalarType::UInt16;
        case 32:
            return ScalarType::UInt32;
        case 64:
            return ScalarType::UInt64;
        default:
            break;
        }
        break;
    case SAMPLEFORMAT_INT:
        switch ( bitsPerSample )
        {
        case 8:
            return ScalarType::Int8;
        case 16:
            return ScalarType::Int16;
        case 32:
            return ScalarType::Int32;
        case 64:
            return ScalarType::Int64;
        default:
            break;
        }
        break;
    case SAMPLEFORMAT_IEEEFP:
        switch ( bitsPerSample )
        {
        case 32:
            return ScalarType::Float32;
        case 64:
            return ScalarType::Float64;
        default:
            break;
        }
        break;
    default:
        break;
    }
    return {};
}

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
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLESPERPIXEL, &res.samplesPerPixel );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_BITSPERSAMPLE, &res.bitsPerSample );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_PLANARCONFIG, &res.planarConfig );
    TIFFGetFieldDefaulted( tiff, TIFFTAG_ORIENTATION, &res.orientation );
    if ( !TIFFGetField( tiff, TIFFTAG_PHOTOMETRIC, &res.photometric ) )
        res.photometric = res.samplesPerPixel >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

    if ( TIFFIsTiled( tiff ) )
    {
        uint32_t tileWidth = 0, tileHeight = 0, tileDepth = 1;
        TIFFGetField( tiff, TIFFTAG_TILEWIDTH, &tileWidth );
        TIFFGetField( tiff, TIFFTAG_TILELENGTH, &tileHeight );
        TIFFGetFieldDefaulted( tiff, TIFFTAG_TILEDEPTH, &tileDepth );
        if ( tileWidth == 0 || tileHeight == 0 || tileDepth > 1 )
            return unexpected( "Unsupported tiles format" );
        res.tileSize = Vector2i( int( tileWidth ), int( tileHeight ) );
    }

    // the samples are read as stored only if they have plain meaning
    bool plainSamples = false;
    switch ( res.photometric )
    {
    case PHOTOMETRIC_MINISWHITE:
    case PHOTOMETRIC_MINISBLACK:
        plainSamples = true;
        break;
    case PHOTOMETRIC_RGB:
        plainSamples = res.samplesPerPixel >= 3;
        break;
    case PHOTOMETRIC_PALETTE:
        plainSamples = res.samplesPerPixel == 1 && ( res.bitsPerSample == 8 || res.bitsPerSample == 16 );
        break;
    default:
        break;
    }
    if ( plainSamples )
        res.sampleType = getSampleType( sampleFormat, res.bitsPerSample );
    if ( !res.sampleType )
    {
        char emsg[1024] = {};
        if ( !TIFFRGBAImageOK( tiff, emsg ) )
            return unexpected( fmt::format( "Unsupported pixel format: {}", emsg ) );
    }
    return res;
}

// returns the values of a tag with variable number of values, e.g. a tag unknown to libtiff
template <typename T>
std::span<const T> getTagValues( TIFF* tiff, uint32_t tag, TIFFDataType type )
{
    const auto* field = TIFFFindField( tiff, tag, type );
    if ( !field || !TIFFFieldPassCount( field ) )
        return {};

    T* data = nullptr;
    size_t count = 0;
    if ( TIFFFieldReadCount( field ) == TIFF_VARIABLE2 )
    {
        uint32_t n = 0;
        if ( !TIFFGetField( tiff, tag, &n, &data ) )
            return {};
        count = n;
    }
    else
    {
        uint16_t n = 0;
        if ( !TIFFGetField( tiff, tag, &n, &data ) )
            return {};
        count = n;
    }
    if ( !data )
        return {};
    return { data, count };
}

std::optional<std::string> getStringTag( TIFF* tiff, uint32_t tag )
{
    const auto* field = TIFFFindField( tiff, tag, TIFF_ASCII );
    if ( !field )
        return {};
    if ( !TIFFFieldPassCount( field ) )
    {
        const char* str = nullptr;
        if ( TIFFGetField( tiff, tag, &str ) && str )
            return std::string( str );
        return {};
    }
    const auto chars = getTagValues<char>( tiff, tag, TIFF_ASCII );
    if ( chars.empty() )
        return {};
    // the count includes the terminating zero
    return std::string( chars.begin(), std::find( chars.begin(), chars.end(), '\0' ) );
}

std::optional<AffineXf3f> readPixelToWorld( TIFF* tiff )
{
    const auto transform = getTagValues<double>( tiff, cModelTransformationTag, TIFF_DOUBLE );
    if ( transform.size() == 16 )
    {
        Matrix4d matrix;
        static_assert( sizeof( matrix ) == 16 * sizeof( double ) );
        std::memcpy( (void*)&matrix, transform.data(), sizeof( matrix ) );
        return AffineXf3f( Matrix4f( matrix ) );
    }

    const auto tiepoint = getTagValues<double>( tiff, cModelTiepointTag, TIFF_DOUBLE );
    const auto pixelScale = getTagValues<double>( tiff, cModelPixelScaleTag, TIFF_DOUBLE );
    if ( tiepoint.size() != 6 || pixelScale.size() != 3 )
        return {};

    Vector3d tiePoints[2] = {
        { tiepoint[0], tiepoint[1], tiepoint[2] },
        { tiepoint[3], tiepoint[4], tiepoint[5] },
    };
    Vector3d scale{ pixelScale[0], pixelScale[1], pixelScale[2] };
    scale.y *= -1.;
    if ( scale.z == 0. )
    {
        tiePoints[1].z = 0.;
        scale.z = 1.;
    }
    return AffineXf3f( Matrix3f::scale( Vector3f( scale ) ), Vector3f( tiePoints[0] + tiePoints[1] ) );
}

// parses a number in the C locale; GDAL also writes nan and inf
std::optional<double> parseNumber( const std::string& str )
{
    const auto first = str.find_first_not_of( " \t" );
    if ( first == std::string::npos )
        return {};
    const auto value = str.substr( first, str.find_last_not_of( " \t" ) + 1 - first );
    const auto lower = toLower( value );
    if ( lower == "nan" || lower == "+nan" || lower == "-nan" )
        return std::numeric_limits<double>::quiet_NaN();
    if ( lower == "inf" || lower == "+inf" )
        return std::numeric_limits<double>::infinity();
    if ( lower == "-inf" )
        return -std::numeric_limits<double>::infinity();

    std::istringstream in( value );
    in.imbue( std::locale::classic() );
    double res = 0;
    in >> res;
    if ( in.fail() || !in.eof() )
        return {};
    return res;
}

std::vector<Color> readPalette( TIFF* tiff, uint16_t bitsPerSample )
{
    uint16_t* red = nullptr;
    uint16_t* green = nullptr;
    uint16_t* blue = nullptr;
    if ( !TIFFGetField( tiff, TIFFTAG_COLORMAP, &red, &green, &blue ) || !red || !green || !blue )
        return {};

    const auto size = size_t( 1 ) << bitsPerSample;
    // as in libtiff's RGBA reader, a color map without values above 255 is an old-style 8-bit one
    bool is16Bit = false;
    for ( size_t i = 0; i < size && !is16Bit; ++i )
        is16Bit = red[i] >= 256 || green[i] >= 256 || blue[i] >= 256;
    const int shift = is16Bit ? 8 : 0;

    std::vector<Color> res( size );
    for ( size_t i = 0; i < size; ++i )
        res[i] = Color( red[i] >> shift, green[i] >> shift, blue[i] >> shift );
    return res;
}

RasterInfo readRasterInfo( TIFF* tiff, const TiffLayout& layout )
{
    RasterInfo res{
        .resolution = layout.size,
    };
    if ( layout.sampleType )
    {
        res.channels = layout.samplesPerPixel;
        res.sampleType = *layout.sampleType;
        if ( layout.photometric == PHOTOMETRIC_PALETTE )
            res.palette = readPalette( tiff, layout.bitsPerSample );
        // as before, only 8-bit and 16-bit samples are inverted, like libtiff's RGBA reader does;
        // writeRawTiff marked all samples with PHOTOMETRIC_MINISWHITE, including floating-point distance maps
        res.minIsWhite = layout.photometric == PHOTOMETRIC_MINISWHITE && layout.bitsPerSample <= 16;
    }
    else
    {
        res.channels = 4;
        res.sampleType = ScalarType::UInt8;
    }
    res.pixelToWorld = readPixelToWorld( tiff );
    if ( const auto noData = getStringTag( tiff, cGdalNoDataTag ) )
        res.noData = parseNumber( *noData );
    return res;
}

// reads the samples as stored in the file to dst, the pixels of the same row and the samples of the same pixel are stored together
Expected<void> readSamples( TIFF* tiff, const TiffLayout& layout, uint8_t* dst, const ProgressCallback& progress )
{
    const auto sampleSize = size_t( layout.bitsPerSample / 8 );
    const auto samplesPerPixel = size_t( layout.samplesPerPixel );
    const auto pixelSize = sampleSize * samplesPerPixel;
    const auto width = size_t( layout.size.x );
    const auto height = size_t( layout.size.y );
    const auto rowSize = width * pixelSize;
    // each sample of a pixel is stored in its own plane
    const bool separatePlanes = layout.planarConfig == PLANARCONFIG_SEPARATE && samplesPerPixel > 1;
    const size_t planeCount = separatePlanes ? samplesPerPixel : 1;
    const size_t planePixelSize = separatePlanes ? sampleSize : pixelSize;

    // copies given number of pixels of a plane to the raster starting from pixel (x, y)
    auto copyPixels = [&] ( const uint8_t* src, size_t plane, size_t x, size_t y, size_t count )
    {
        auto* d = dst + y * rowSize + x * pixelSize;
        if ( !separatePlanes )
        {
            std::memcpy( d, src, count * pixelSize );
            return;
        }
        d += plane * sampleSize;
        for ( size_t i = 0; i < count; ++i )
            std::memcpy( d + i * pixelSize, src + i * sampleSize, sampleSize );
    };

    if ( layout.tileSize )
    {
        const auto tileWidth = size_t( layout.tileSize->x );
        const auto tileHeight = size_t( layout.tileSize->y );
        const auto tileRowSize = tileWidth * planePixelSize;
        Buffer<uint8_t> buffer( tileRowSize * tileHeight );
        const auto tileCount = planeCount * ( ( width + tileWidth - 1 ) / tileWidth ) * ( ( height + tileHeight - 1 ) / tileHeight );
        size_t tilesRead = 0;
        for ( size_t plane = 0; plane < planeCount; ++plane )
        {
            for ( size_t y = 0; y < height; y += tileHeight )
            {
                for ( size_t x = 0; x < width; x += tileWidth )
                {
                    const auto tile = TIFFComputeTile( tiff, uint32_t( x ), uint32_t( y ), 0, uint16_t( plane ) );
                    if ( TIFFReadEncodedTile( tiff, tile, buffer.data(), tmsize_t( buffer.size() ) ) < 0 )
                        return unexpected( "Error reading tile" );
                    // the tiles of the last column and of the last row can extend beyond the image
                    const auto rows = std::min( tileHeight, height - y );
                    const auto columns = std::min( tileWidth, width - x );
                    for ( size_t row = 0; row < rows; ++row )
                        copyPixels( buffer.data() + row * tileRowSize, plane, x, y + row, columns );
                    if ( !reportProgress( progress, float( ++tilesRead ) / float( tileCount ) ) )
                        return unexpectedOperationCanceled();
                }
            }
        }
    }
    else
    {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted( tiff, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip );
        const auto stripRows = std::clamp( size_t( rowsPerStrip ), size_t( 1 ), height );
        const auto planeRowSize = width * planePixelSize;
        Buffer<uint8_t> buffer;
        if ( separatePlanes )
            buffer.resize( planeRowSize * stripRows );
        const auto stripCount = planeCount * ( ( height + stripRows - 1 ) / stripRows );
        size_t stripsRead = 0;
        for ( size_t plane = 0; plane < planeCount; ++plane )
        {
            for ( size_t y = 0; y < height; y += stripRows )
            {
                const auto rows = std::min( stripRows, height - y );
                const auto strip = TIFFComputeStrip( tiff, uint32_t( y ), uint16_t( plane ) );
                // the strips of contiguous samples are read directly to the raster
                auto* stripData = separatePlanes ? buffer.data() : dst + y * rowSize;
                if ( TIFFReadEncodedStrip( tiff, strip, stripData, tmsize_t( rows * planeRowSize ) ) < 0 )
                    return unexpected( "Error reading strip" );
                if ( separatePlanes )
                {
                    for ( size_t row = 0; row < rows; ++row )
                        copyPixels( buffer.data() + row * planeRowSize, plane, 0, y + row, width );
                }
                if ( !reportProgress( progress, float( ++stripsRead ) / float( stripCount ) ) )
                    return unexpectedOperationCanceled();
            }
        }
    }
    return {};
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

Expected<Raster> readRaster( TIFF* tiff, const ProgressCallback& progress, TiffLayout* outLayout )
{
    auto layout = readLayout( tiff );
    if ( !layout )
        return unexpected( std::move( layout.error() ) );
    if ( outLayout )
        *outLayout = *layout;

    Raster res{
        .info = readRasterInfo( tiff, *layout ),
    };
    res.data.resize( res.info.dataSize() );
    if ( layout->sampleType )
    {
        if ( auto readRes = readSamples( tiff, *layout, res.data.data(), progress ); !readRes )
            return unexpected( std::move( readRes.error() ) );
        applyOrientation( res.data.data(), layout->size, size_t( res.info.channels ) * res.info.sampleSize(), layout->orientation );
    }
    else
    {
        // TIFFReadRGBAImageOriented stores the pixels as uint32 values with the red component in the lowest byte,
        // which is the order of the samples in a little-endian machine
        static_assert( std::endian::native == std::endian::little );
        if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( layout->size.x ), uint32_t( layout->size.y ), (uint32_t*)res.data.data(), ORIENTATION_TOPLEFT, 1 ) )
            return unexpected( "Error reading pixels" );
        if ( !reportProgress( progress, 1.f ) )
            return unexpectedOperationCanceled();
    }
    return res;
}

Expected<Raster> loadRaster( const std::filesystem::path& path, const ProgressCallback& progress, TiffLayout* outLayout = nullptr )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    return addFileName( readRaster( tiff, progress, outLayout ), path );
}

// leaves only the first sample of each pixel
void keepFirstChannel( Raster& raster )
{
    const auto sampleSize = raster.info.sampleSize();
    const auto pixelSize = sampleSize * size_t( raster.info.channels );
    const auto pixelCount = size_t( raster.info.resolution.x ) * size_t( raster.info.resolution.y );
    for ( size_t i = 1; i < pixelCount; ++i )
        std::memmove( raster.data.data() + i * sampleSize, raster.data.data() + i * pixelSize, sampleSize );
    raster.data.resize( pixelCount * sampleSize );
    raster.info.channels = 1;
}

// writes a raster with given properties and samples, getRow returns the samples of given row, the rows go from top to bottom
Expected<void> writeTiff( const std::filesystem::path& path, const RasterInfo& info, const std::function<const uint8_t* ( size_t )>& getRow,
    const ProgressCallback& progress )
{
    const auto sampleSize = info.sampleSize();
    if ( sampleSize == 0 )
        return unexpected( "Unsupported sample type" );
    if ( info.resolution.x <= 0 || info.resolution.y <= 0 || info.channels <= 0 )
        return unexpected( "Cannot save empty raster" );

    int sampleFormat = SAMPLEFORMAT_UINT;
    switch ( info.sampleType )
    {
    case ScalarType::Int8:
    case ScalarType::Int16:
    case ScalarType::Int32:
    case ScalarType::Int64:
        sampleFormat = SAMPLEFORMAT_INT;
        break;
    case ScalarType::Float32:
    case ScalarType::Float64:
        sampleFormat = SAMPLEFORMAT_IEEEFP;
        break;
    default:
        break;
    }

    const bool hasPalette = info.channels == 1 && !info.palette.empty();
    if ( hasPalette && info.sampleType != ScalarType::UInt8 && info.sampleType != ScalarType::UInt16 )
        return unexpected( "Palette requires 8-bit or 16-bit unsigned samples" );

    TIFF* tiff = openTiff( path, "w" );
    if ( !tiff )
        return unexpected( "Cannot write file: " + utf8string( path ) );
    MR_FINALLY {
        TIFFClose( tiff );
    };

    TIFFSetField( tiff, TIFFTAG_IMAGEWIDTH, uint32_t( info.resolution.x ) );
    TIFFSetField( tiff, TIFFTAG_IMAGELENGTH, uint32_t( info.resolution.y ) );
    TIFFSetField( tiff, TIFFTAG_BITSPERSAMPLE, int( sampleSize * 8 ) );
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, info.channels );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sampleFormat );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );

    if ( hasPalette )
    {
        TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_PALETTE );
        // the color map has an entry for each possible sample value, its components are 16-bit
        const auto size = size_t( 1 ) << ( sampleSize * 8 );
        std::vector<uint16_t> red( size ), green( size ), blue( size );
        for ( size_t i = 0; i < std::min( size, info.palette.size() ); ++i )
        {
            red[i] = uint16_t( info.palette[i].r * 257 );
            green[i] = uint16_t( info.palette[i].g * 257 );
            blue[i] = uint16_t( info.palette[i].b * 257 );
        }
        TIFFSetField( tiff, TIFFTAG_COLORMAP, red.data(), green.data(), blue.data() );
    }
    else if ( info.channels <= 2 )
    {
        TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, info.minIsWhite ? PHOTOMETRIC_MINISWHITE : PHOTOMETRIC_MINISBLACK );
    }
    else
    {
        TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB );
    }

    // the channel after the gray or RGB ones is alpha, the next ones have no predefined meaning
    const int colorChannels = info.channels <= 2 ? 1 : 3;
    if ( info.channels > colorChannels )
    {
        std::vector<uint16_t> extraSamples( size_t( info.channels - colorChannels ), EXTRASAMPLE_UNSPECIFIED );
        extraSamples.front() = EXTRASAMPLE_UNASSALPHA;
        TIFFSetField( tiff, TIFFTAG_EXTRASAMPLES, int( extraSamples.size() ), extraSamples.data() );
    }

    // declare non-standard tags
    std::vector<TIFFFieldInfo> fieldInfo;
    if ( info.pixelToWorld )
        fieldInfo.push_back( { cModelTransformationTag, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTransformationTag" } );
    if ( info.noData )
        fieldInfo.push_back( { cGdalNoDataTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, (char*)"GDALNoDataValue" } );
    if ( !fieldInfo.empty() )
        TIFFMergeFieldInfo( tiff, fieldInfo.data(), uint32_t( fieldInfo.size() ) );

    if ( info.pixelToWorld )
    {
        const Matrix4d matrix = AffineXf3d( *info.pixelToWorld );
        TIFFSetField( tiff, cModelTransformationTag, 16, &matrix );
    }
    if ( info.noData )
        TIFFSetField( tiff, cGdalNoDataTag, fmt::format( "{}", *info.noData ).c_str() );

    const auto height = size_t( info.resolution.y );
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
    return loadRaster( path, settings.progress );
}

Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    auto layout = readLayout( tiff );
    if ( !layout )
        return addFileName( Expected<RasterInfo>( unexpected( std::move( layout.error() ) ) ), path );

    return readRasterInfo( tiff, *layout );
}

MR_ADD_RASTER_LOADER( IOFilter( "TIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff, infoFromTiff )

} // namespace RasterLoad

namespace RasterSave
{

Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings )
{
    MR_TIMER;
    if ( raster.data.size() != raster.info.dataSize() )
        return unexpected( "Raster data size does not match its resolution" );

    const auto rowSize = size_t( raster.info.resolution.x ) * size_t( raster.info.channels ) * raster.info.sampleSize();
    return writeTiff( path, raster.info, [&] ( size_t row ) { return raster.data.data() + row * rowSize; }, settings.progress );
}

// FIXME: single filter
MR_ADD_RASTER_SAVER( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff )
MR_ADD_RASTER_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace RasterSave

namespace ImageLoad
{

Expected<Image> fromTiff( const std::filesystem::path& path )
{
    TiffLayout layout;
    auto raster = loadRaster( path, {}, &layout );
    if ( !raster )
        return unexpected( std::move( raster.error() ) );

    // as in libtiff's RGBA reader, a gray image with more than two samples per pixel shows only the first one
    const bool isGray = layout.photometric == PHOTOMETRIC_MINISBLACK || layout.photometric == PHOTOMETRIC_MINISWHITE;
    if ( layout.sampleType && isGray && raster->info.channels > 2 )
        keepFirstChannel( *raster );

    return convertRasterToImage( *raster );
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
        .resolution = image.resolution,
        .channels = 4,
        .sampleType = ScalarType::UInt8,
    };
    const auto width = size_t( image.resolution.x );
    const auto height = size_t( image.resolution.y );
    // Image starts from the bottom row
    return writeTiff( path, info, [&] ( size_t row ) { return (const uint8_t*)( image.pixels.data() + ( height - 1 - row ) * width ); }, {} );
}

// FIXME: single filter
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff, -1 )
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff, -1 )

} // namespace ImageSave

namespace DistanceMapSave
{

Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings )
{
    RasterInfo info{
        .resolution = dmap.dims(),
        .sampleType = ScalarType::Float32,
        .noData = DistanceMap::NOT_VALID_VALUE,
    };
    if ( settings.xf )
        info.pixelToWorld = *settings.xf;

    const auto width = dmap.resX();
    return writeTiff( path, info, [&] ( size_t row ) { return (const uint8_t*)( dmap.data() + row * width ); }, settings.progress );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff )

} // namespace DistanceMapSave

} // namespace MR
#endif
