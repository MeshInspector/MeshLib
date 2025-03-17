#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRFmt.h"

#include <tiffio.h>

namespace
{
using namespace MR;

struct TiffParameters
{
    enum class SampleType
    {
        Unknown = 0,
        UInt = SAMPLEFORMAT_UINT,
        Int = SAMPLEFORMAT_INT,
        Float = SAMPLEFORMAT_IEEEFP,
    } sampleType{ SampleType::Unknown };

    enum class ValueType
    {
        Unknown = 0,
        Scalar = 1,
        RGB = 3,
        RGBA = 4,
    } valueType{ ValueType::Unknown };

    // size of internal data in file
    int bytesPerSample = 0;

    // size of image if not layered, otherwise size of layer
    Vector2i imageSize;

    // size of tile if tiled, otherwise zero
    std::optional<Vector2i> tileSize;

    bool operator==( const TiffParameters& ) const = default;
};

class TiffHolder
{
public:
    TiffHolder( const std::filesystem::path& path, const char* mode )
    {
#ifdef __WIN32__
        tiffPtr_ = TIFFOpenW( path.wstring().c_str(), mode );
#else
        tiffPtr_ = TIFFOpen( utf8string( path ).c_str(), mode );
#endif
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

TiffParameters readTiffParameters( TIFF* tiff )
{
    TiffParameters params;

    TIFFGetField( tiff, TIFFTAG_BITSPERSAMPLE, &params.bytesPerSample );
    params.bytesPerSample >>= 3; // convert to bytes

    int samplePerPixel = 0;
    TIFFGetField( tiff, TIFFTAG_SAMPLESPERPIXEL, &samplePerPixel );
    if ( samplePerPixel == 0 )
    {
        // incorrect tiff format, treat like Scalar
        samplePerPixel = 1;
    }
    params.valueType = TiffParameters::ValueType( samplePerPixel );

    int sampleFormat = 0;
    TIFFGetField( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    if ( sampleFormat == 0 )
    {
        // incorrect tiff format, treat like UInt
        sampleFormat = SAMPLEFORMAT_UINT;
    }
    params.sampleType = TiffParameters::SampleType( sampleFormat );

    TIFFGetField( tiff, TIFFTAG_IMAGEWIDTH, &params.imageSize.x );
    TIFFGetField( tiff, TIFFTAG_IMAGELENGTH, &params.imageSize.y );

    if ( TIFFIsTiled( tiff ) )
    {
        params.tileSize.emplace();
        TIFFGetField( tiff, TIFFTAG_TILEWIDTH, &params.tileSize->x );
        TIFFGetField( tiff, TIFFTAG_TILELENGTH, &params.tileSize->y );
    }

    return params;
}

template <typename T, typename U>
T rgbToScalar( const U* src )
{
    // luma/brightness component from the YCbCr color space
    return T(
          (float)src[0] * 0.299f
        + (float)src[1] * 0.587f
        + (float)src[2] * 0.114f
    );
}

template <typename U>
void copyFromTiffImpl( Color* dst, const U* src, size_t size, const TiffParameters& tp )
{
    switch ( tp.valueType )
    {
    case TiffParameters::ValueType::RGBA:
        if constexpr ( std::is_same_v<uint8_t, U> )
        {
            std::copy( (Color*)src, (Color*)src + size, dst );
        }
        else
        {
            for ( auto i = 0u; i < size; ++i )
                for ( auto j = 0; j < 4; ++j )
                    dst[i][j] = Color::valToUint8( src[i * 4 + j] );
        }
        break;
    case TiffParameters::ValueType::RGB:
        for ( auto i = 0u; i < size; ++i )
            for ( auto j = 0; j < 3; ++j )
                dst[i][j] = Color::valToUint8( src[i * 3 + j] );
        break;
    case TiffParameters::ValueType::Scalar:
        for ( auto i = 0u; i < size; ++i )
            for ( auto j = 0; j < 3; ++j )
                dst[i][j] = Color::valToUint8( src[i] );
        break;
    case TiffParameters::ValueType::Unknown:
        MR_UNREACHABLE_NO_RETURN;
        break;
    }
}

template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void copyFromTiffImpl( T* dst, const U* src, size_t size, const TiffParameters& tp )
{
    switch ( tp.valueType )
    {
    case TiffParameters::ValueType::Scalar:
        std::copy( src, src + size, dst );
        break;
    case TiffParameters::ValueType::RGB:
        for ( auto i = 0u; i < size; ++i )
            dst[i] = rgbToScalar<T>( src + i * 3 );
        break;
    case TiffParameters::ValueType::RGBA:
        for ( auto i = 0u; i < size; ++i )
            dst[i] = rgbToScalar<T>( src + i * 4 );
        break;
    case TiffParameters::ValueType::Unknown:
        MR_UNREACHABLE_NO_RETURN;
        break;
    }
}

template <typename Func>
void visitTiffData( Func f, const uint8_t* data, const TiffParameters& tp )
{
#define CALL_IF( Type ) \
    if ( tp.bytesPerSample == sizeof( Type ) ) \
        return f( reinterpret_cast<const Type*>( data ) );

    switch ( tp.sampleType )
    {
    case TiffParameters::SampleType::UInt:
        CALL_IF( uint8_t )
        CALL_IF( uint16_t )
        CALL_IF( uint32_t )
        CALL_IF( uint64_t )
        MR_UNREACHABLE_NO_RETURN
        break;
    case TiffParameters::SampleType::Int:
        CALL_IF( int8_t )
        CALL_IF( int16_t )
        CALL_IF( int32_t )
        CALL_IF( int64_t )
        MR_UNREACHABLE_NO_RETURN
        break;
    case TiffParameters::SampleType::Float:
        CALL_IF( float )
        CALL_IF( double )
        MR_UNREACHABLE_NO_RETURN
        break;
    case TiffParameters::SampleType::Unknown:
        MR_UNREACHABLE_NO_RETURN
        break;
    }

#undef CALL_IF
}

template <typename T>
void readTiff( TIFF* tiff, T* bytes, [[maybe_unused]] size_t size, const TiffParameters& tp )
{
    assert( tp.bytesPerSample != 0 );
    assert( (int)tp.valueType != 0 );
    assert( (size_t)tp.imageSize.x * tp.imageSize.y >= size );
    const auto pixelSize = (size_t)tp.valueType * tp.bytesPerSample;

    Buffer<uint8_t> buffer;
    if ( tp.tileSize )
    {
        buffer.resize( (size_t)tp.tileSize->x * tp.tileSize->y * pixelSize );

        for ( auto tileY : splitByChunks( tp.imageSize.y, tp.tileSize->y ) )
        {
            for ( auto tileX : splitByChunks( tp.imageSize.x, tp.tileSize->x ) )
            {
                TIFFReadTile( tiff, buffer.data(), tileX.offset, tileY.offset, 0, 0 );

                for ( auto tileRow = 0u; tileRow < tileY.size; ++tileRow )
                {
                    const auto imageOffset = tileX.offset + ( tileY.offset + tileRow ) * tp.imageSize.x;
                    const auto tileOffset = tileRow * tileX.size;
                    visitTiffData( [&] <typename U> ( const U* data )
                    {
                        copyFromTiffImpl( bytes + imageOffset, data, tileX.size, tp );
                    }, buffer.data() + tileOffset * pixelSize, tp );
                }
            }
        }
    }
    else
    {
        buffer.resize( (size_t)tp.imageSize.x * pixelSize );

        for ( auto row = 0u; row < tp.imageSize.y; ++row )
        {
            TIFFReadScanline( tiff, buffer.data(), row );

            const auto offset = (size_t)row * tp.imageSize.x;
            visitTiffData( [&] <typename U> ( const U* data )
            {
                copyFromTiffImpl( bytes + offset, data, tp.imageSize.x, tp );
            }, buffer.data(), tp );
        }
    }
}

} // namespace

namespace MR
{

namespace ImageLoad
{

Expected<Image> fromTiff( const std::filesystem::path& path )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    auto params = readTiffParameters( tiff );

    Image result {
        .resolution = params.imageSize,
    };
    result.pixels.resize( (size_t)params.imageSize.x * params.imageSize.y );

    char emsg[21024];
    if ( TIFFRGBAImageOK( tiff, emsg ) )
    {
        TIFFReadRGBAImageOriented( tiff, result.resolution.x, result.resolution.y, (uint32_t*)result.pixels.data(), ORIENTATION_TOPLEFT, 0 );
    }
    else if ( params.valueType == TiffParameters::ValueType::RGB || params.valueType == TiffParameters::ValueType::RGBA )
    {
        readTiff( tiff, result.pixels.data(), result.pixels.size(), params );
    }
    else
    {
        assert( params.valueType == TiffParameters::ValueType::Scalar );
        visitTiffData( [&] <typename U> ( const U* )
        {
            Buffer<U> buffer;
            buffer.resize( result.pixels.size() );

            readTiff( tiff, buffer.data(), buffer.size(), params );

            const auto [min, max] = parallelMinMax( buffer.data(), buffer.size() );
            ParallelFor( (size_t)0, result.pixels.size(), [&] ( size_t i )
            {
                const auto value = (uint8_t)( 255. * (double)( buffer[i] - min ) / (double)( max - min ) );
                result.pixels[i] = { value, value, value };
            } );
        }, nullptr, params );
    }

    return result;
}

MR_ADD_IMAGE_LOADER_WITH_PRIORITY( IOFilter( "TIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff, -1 )

} // namespace ImageLoad

namespace ImageSave
{

Expected<void> toTiff( const Image& image, const std::filesystem::path& path )
{
    auto tiff = TIFFOpen( utf8string( path ).c_str(), "w" );
    if ( !tiff )
        return unexpected( "Cannot write file: " + utf8string( path ) );
    MR_FINALLY {
        TIFFClose( tiff );
    };

    TIFFSetField( tiff, TIFFTAG_IMAGEWIDTH, image.resolution.x );
    TIFFSetField( tiff, TIFFTAG_IMAGELENGTH, image.resolution.y );

    // 32-bit RGBA
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT );
    TIFFSetField( tiff, TIFFTAG_BITSPERSAMPLE, 8 );
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, 4 );

    // FIXME: orientation is ignored
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );

    for ( auto row = 0u; row < image.resolution.y; ++row )
    {
        // FIXME: orientation is ignored
        const auto* data = image.pixels.data() + ( image.resolution.y - 1 - row ) * image.resolution.x;
        TIFFWriteScanline( tiff, (void*)data, row );
    }
    TIFFFlush( tiff );

    return {};
}

// FIXME: single filter
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tif)", "*.tif" ), toTiff, -1 )
MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "TIFF (.tiff)", "*.tiff" ), toTiff, -1 )

} // namespace ImageSave

} // namespace MR
#endif
