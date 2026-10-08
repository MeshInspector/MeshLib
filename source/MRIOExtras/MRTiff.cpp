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

namespace
{
using namespace MR;

// GeoTIFF tag: http://geotiff.maptools.org/spec/geotiff2.6.html
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

// how the stored samples become the values of the raster
enum class Conversion
{
    // the first sample of each pixel is taken as is
    FirstSample,
    // gray and alpha samples become RGBA8 values
    GrayAlpha,
    // the first three samples become RGB8 values
    Rgb,
    // the first four samples become RGBA8 values
    Rgba,
    // palette indices become RGB8 values
    Palette,
    // libtiff decodes the pixels to RGBA8 values
    Decoded,
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

Conversion getConversion( const TiffLayout& layout )
{
    if ( !layout.sampleType )
        return Conversion::Decoded;
    switch ( layout.photometric )
    {
    case PHOTOMETRIC_RGB:
        return layout.samplesPerPixel == 3 ? Conversion::Rgb : Conversion::Rgba;
    case PHOTOMETRIC_PALETTE:
        return Conversion::Palette;
    default:
        // as in libtiff's RGBA reader, the second of two gray samples is alpha, and only the first sample is shown otherwise
        if ( layout.samplesPerPixel == 2 && ( *layout.sampleType == ScalarType::UInt8 || *layout.sampleType == ScalarType::UInt16 ) )
            return Conversion::GrayAlpha;
        return Conversion::FirstSample;
    }
}

ScalarType getValueType( const TiffLayout& layout, Conversion conversion )
{
    switch ( conversion )
    {
    case Conversion::FirstSample:
        return *layout.sampleType;
    case Conversion::Rgb:
    case Conversion::Palette:
        return ScalarType::RGB8;
    case Conversion::GrayAlpha:
    case Conversion::Rgba:
    case Conversion::Decoded:
        return ScalarType::RGBA8;
    }
    MR_UNREACHABLE
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
                // the strips of contiguous samples are read directly to the destination
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

// calls f( T{} ) with T being the type of the stored samples
template <typename F>
void visitSampleType( ScalarType sampleType, F&& f )
{
    switch ( sampleType )
    {
    case ScalarType::UInt8:
        return f( uint8_t{} );
    case ScalarType::Int8:
        return f( int8_t{} );
    case ScalarType::UInt16:
        return f( uint16_t{} );
    case ScalarType::Int16:
        return f( int16_t{} );
    case ScalarType::UInt32:
        return f( uint32_t{} );
    case ScalarType::Int32:
        return f( int32_t{} );
    case ScalarType::UInt64:
        return f( uint64_t{} );
    case ScalarType::Int64:
        return f( int64_t{} );
    case ScalarType::Float32:
        return f( float{} );
    case ScalarType::Float64:
        return f( double{} );
    default:
        MR_UNREACHABLE_NO_RETURN
    }
}

template <typename T>
T getSample( const uint8_t* data, size_t index )
{
    T res;
    std::memcpy( &res, data + index * sizeof( T ), sizeof( T ) );
    return res;
}

// converts a color or alpha sample to 8 bits, 16-bit samples are rounded as libtiff's RGBA reader does
template <typename T>
uint8_t colorTo8Bit( T v )
{
    if constexpr ( std::is_same_v<T, uint16_t> )
        return uint8_t( ( v + 128u ) / 257u );
    else
        return Color::valToUint8( v );
}

// converts the samples as stored in the file to the values of the raster
void convertSamples( const TiffLayout& layout, Conversion conversion, const std::vector<Color>& palette, const uint8_t* src, uint8_t* dst )
{
    const auto pixelCount = size_t( layout.size.x ) * size_t( layout.size.y );
    const auto samplesPerPixel = size_t( layout.samplesPerPixel );
    const bool minIsWhite = layout.photometric == PHOTOMETRIC_MINISWHITE;
    visitSampleType( *layout.sampleType, [&] <typename T> ( T )
    {
        auto sample = [src] ( size_t i ) { return getSample<T>( src, i ); };
        switch ( conversion )
        {
        case Conversion::FirstSample:
            ParallelFor( size_t( 0 ), pixelCount, [&] ( size_t i )
            {
                std::memcpy( dst + i * sizeof( T ), src + i * samplesPerPixel * sizeof( T ), sizeof( T ) );
            } );
            break;
        case Conversion::GrayAlpha:
            if constexpr ( std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> )
            {
                ParallelFor( size_t( 0 ), pixelCount, [&] ( size_t i )
                {
                    // the gray value is the high byte, as in libtiff's RGBA reader
                    auto gray = uint8_t( sample( 2 * i ) >> ( 8 * ( sizeof( T ) - 1 ) ) );
                    if ( minIsWhite )
                        gray = uint8_t( 255 - gray );
                    auto* d = dst + i * 4;
                    d[0] = d[1] = d[2] = gray;
                    d[3] = colorTo8Bit( sample( 2 * i + 1 ) );
                } );
            }
            break;
        case Conversion::Rgb:
        case Conversion::Rgba:
        {
            const size_t channels = conversion == Conversion::Rgb ? 3 : 4;
            ParallelFor( size_t( 0 ), pixelCount, [&] ( size_t i )
            {
                for ( size_t c = 0; c < channels; ++c )
                    dst[i * channels + c] = colorTo8Bit( sample( i * samplesPerPixel + c ) );
            } );
            break;
        }
        case Conversion::Palette:
            if constexpr ( std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> )
            {
                ParallelFor( size_t( 0 ), pixelCount, [&] ( size_t i )
                {
                    const auto index = size_t( sample( i ) );
                    const auto c = index < palette.size() ? palette[index] : Color::black();
                    auto* d = dst + i * 3;
                    d[0] = c.r;
                    d[1] = c.g;
                    d[2] = c.b;
                } );
            }
            break;
        case Conversion::Decoded:
            MR_UNREACHABLE_NO_RETURN
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

// how a raster was read from the file
struct ReadInfo
{
    TiffLayout layout;
    Conversion conversion = Conversion::Decoded;
};

Expected<Raster> readRaster( TIFF* tiff, const ProgressCallback& progress, ReadInfo* outReadInfo )
{
    auto layout = readLayout( tiff );
    if ( !layout )
        return unexpected( std::move( layout.error() ) );
    const auto conversion = getConversion( *layout );
    if ( outReadInfo )
        *outReadInfo = { *layout, conversion };

    Raster res{
        .info = {
            .dims = Vector3i( layout->size.x, layout->size.y, 1 ),
            .type = getValueType( *layout, conversion ),
        },
    };
    res.data.resize( res.info.dataSize() );

    if ( conversion == Conversion::Decoded )
    {
        // TIFFReadRGBAImageOriented stores the pixels as uint32 values with the red component in the lowest byte,
        // which is the order of the components in a little-endian machine
        static_assert( std::endian::native == std::endian::little );
        if ( !TIFFReadRGBAImageOriented( tiff, uint32_t( layout->size.x ), uint32_t( layout->size.y ), (uint32_t*)res.data.data(), ORIENTATION_TOPLEFT, 1 ) )
            return unexpected( "Error reading pixels" );
        if ( !reportProgress( progress, 1.f ) )
            return unexpectedOperationCanceled();
        return res;
    }

    // the samples are read directly to the raster if they need no conversion
    const auto storedPixelSize = size_t( layout->samplesPerPixel ) * size_t( layout->bitsPerSample / 8 );
    const bool isColor = conversion == Conversion::Rgb || conversion == Conversion::Rgba;
    const bool asStored = ( conversion == Conversion::FirstSample && layout->samplesPerPixel == 1 )
        || ( isColor && *layout->sampleType == ScalarType::UInt8 && storedPixelSize == getScalarTypeSize( res.info.type ) );
    if ( asStored )
    {
        if ( auto readRes = readSamples( tiff, *layout, res.data.data(), progress ); !readRes )
            return unexpected( std::move( readRes.error() ) );
    }
    else
    {
        Buffer<uint8_t> stored( size_t( layout->size.x ) * size_t( layout->size.y ) * storedPixelSize );
        if ( auto readRes = readSamples( tiff, *layout, stored.data(), progress ); !readRes )
            return unexpected( std::move( readRes.error() ) );
        const auto palette = conversion == Conversion::Palette ? readPalette( tiff, layout->bitsPerSample ) : std::vector<Color>{};
        convertSamples( *layout, conversion, palette, stored.data(), res.data.data() );
    }
    applyOrientation( res.data.data(), layout->size, getScalarTypeSize( res.info.type ), layout->orientation );
    return res;
}

Expected<Raster> loadRaster( const std::filesystem::path& path, const ProgressCallback& progress, ReadInfo* outReadInfo = nullptr )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    return addFileName( readRaster( tiff, progress, outReadInfo ), path );
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
    int samplesPerPixel = 1;
    int sampleFormat = SAMPLEFORMAT_UINT;
    int photometric = PHOTOMETRIC_MINISBLACK;
    const auto valueSize = getScalarTypeSize( info.type );
    int bitsPerSample = int( valueSize * 8 );
    switch ( info.type )
    {
    case ScalarType::UInt8:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
        break;
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
    case ScalarType::RGB8:
    case ScalarType::RGBA8:
        samplesPerPixel = int( valueSize );
        bitsPerSample = 8;
        photometric = PHOTOMETRIC_RGB;
        break;
    case ScalarType::Float32_4:
    case ScalarType::Unknown:
    case ScalarType::Count:
        return unexpected( "Unsupported value type" );
    }
    if ( info.dims.x <= 0 || info.dims.y <= 0 || info.dims.z <= 0 )
        return unexpected( "Cannot save empty raster" );
    if ( info.dims.z != 1 )
        return unexpected( "Cannot save a raster with several layers" );

    TIFF* tiff = openTiff( path, "w" );
    if ( !tiff )
        return unexpected( "Cannot write file: " + utf8string( path ) );
    MR_FINALLY {
        TIFFClose( tiff );
    };

    TIFFSetField( tiff, TIFFTAG_IMAGEWIDTH, uint32_t( info.dims.x ) );
    TIFFSetField( tiff, TIFFTAG_IMAGELENGTH, uint32_t( info.dims.y ) );
    TIFFSetField( tiff, TIFFTAG_BITSPERSAMPLE, bitsPerSample );
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, sampleFormat );
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, photometric );
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

    return RasterInfo{
        .dims = Vector3i( layout->size.x, layout->size.y, 1 ),
        .type = getValueType( *layout, getConversion( *layout ) ),
    };
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
    ReadInfo readInfo;
    auto raster = loadRaster( path, {}, &readInfo );
    if ( !raster )
        return unexpected( std::move( raster.error() ) );

    auto res = convertRasterToImage( *raster );
    // as in libtiff's RGBA reader, 8-bit and 16-bit gray values are inverted if the smallest value means white;
    // other gray values are not, since writeRawTiff marked all files including floating-point distance maps this way
    const auto& layout = readInfo.layout;
    if ( res && readInfo.conversion == Conversion::FirstSample && layout.photometric == PHOTOMETRIC_MINISWHITE && layout.bitsPerSample <= 16 )
    {
        for ( auto& c : res->pixels )
            c = Color( 255 - c.r, 255 - c.g, 255 - c.b, c.a );
    }
    return res;
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
