#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRChunkIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRFmt.h"

#include <tiffio.h>

namespace
{

using namespace MR;

enum class DataType
{
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
};

template <typename Func>
auto visitData( const std::byte* data, DataType type, Func&& f )
{
#define CALL_IF( enumValue, TypeName ) case DataType::enumValue: return f( reinterpret_cast<const TypeName*>( data ) );
    switch ( type )
    {
        CALL_IF( UInt8, uint8_t )
        CALL_IF( UInt16, uint16_t )
        CALL_IF( UInt32, uint32_t )
        CALL_IF( UInt64, uint64_t )
        CALL_IF( Int8, int8_t )
        CALL_IF( Int16, int16_t )
        CALL_IF( Int32, int32_t )
        CALL_IF( Int64, int64_t )
        CALL_IF( Float32, float )
        CALL_IF( Float64, double )
    }
#undef CALL_IF
}

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

    [[nodiscard]] size_t getPixelSize() const
    {
        return (size_t)valueType * bytesPerSample;
    }

    [[nodiscard]] DataType getDataType() const
    {
        switch ( sampleType )
        {
        case SampleType::UInt:
            switch ( bytesPerSample )
            {
            case 1: return DataType::UInt8;
            case 2: return DataType::UInt16;
            case 4: return DataType::UInt32;
            case 8: return DataType::UInt64;
            default: MR_UNREACHABLE;
            }
        case SampleType::Int:
            switch ( bytesPerSample )
            {
            case 1: return DataType::Int8;
            case 2: return DataType::Int16;
            case 4: return DataType::Int32;
            case 8: return DataType::Int64;
            default: MR_UNREACHABLE;
            }
        case SampleType::Float:
            switch ( bytesPerSample )
            {
            case 4: return DataType::Float32;
            case 8: return DataType::Float64;
            default: MR_UNREACHABLE;
            }
        case SampleType::Unknown:
            MR_UNREACHABLE
        }
        MR_UNREACHABLE
    }
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

// http://geotiff.maptools.org/spec/geotiff2.6.html
enum GeoTiff : uint32_t
{
    ModelTiepointTag = 33922,
    ModelPixelScaleTag = 33550,
    ModelTransformationTag = 34264,
};

std::optional<AffineXf3f> readGeoTiffParameters( TIFF* tiff )
{
    Matrix4d matrix;
    if ( TIFFGetField( tiff, GeoTiff::ModelTransformationTag, &matrix ) )
        return AffineXf3f { Matrix4f( matrix ) };

    double* dataTiepoint, * dataScale; // will be freed with tiff
    uint32_t count;
    if ( !TIFFGetField( tiff, GeoTiff::ModelTiepointTag, &count, &dataTiepoint ) || count != 6 )
        return {};
    if ( !TIFFGetField( tiff, GeoTiff::ModelPixelScaleTag, &count, &dataScale ) || count != 3 )
        return {};

    std::array tiepoints {
        Vector3d { dataTiepoint[0], dataTiepoint[1], dataTiepoint[2] },
        Vector3d { dataTiepoint[3], dataTiepoint[4], dataTiepoint[5] },
    };
    Vector3d scale { dataScale[0], dataScale[1], dataScale[2] };

    // TODO: explain the operations
    scale.y *= -1.;
    if ( scale.z == 0. )
    {
        tiepoints[0].z = 0.;
        scale.z = 1.;
    }

    return AffineXf3f {
        Matrix3f::scale( Vector3f( scale ) ),
        Vector3f( tiepoints[0] + tiepoints[1] )
    };
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
Color scalarToRgb( U src )
{
    Color result;
    for ( auto i = 0; i < 3; ++i )
        result[i] = Color::valToUint8( src );
    return result;
}

template <typename U>
Color scalarToRgb( U src, U min, U max )
{
    return scalarToRgb( (float)( src - min ) / (float)( max - min ) );
}

template <typename U>
Color scalarToRgb( const U* src, size_t sampleCount )
{
    Color result;
    for ( auto i = 0; i < sampleCount; ++i )
        result[i] = Color::valToUint8( src[i] );
    return result;
}

template <typename U>
void copySampledData( Color* dst, const U* src, size_t size, TiffParameters::ValueType valueType )
{
    switch ( valueType )
    {
    case TiffParameters::ValueType::RGBA:
        if constexpr ( std::is_same_v<uint8_t, U> )
        {
            std::copy_n( (Color*)src, size, dst );
        }
        else
        {
            for ( auto i = 0u; i < size; ++i )
                dst[i] = scalarToRgb( src + i * 4, 4 );
        }
        break;
    case TiffParameters::ValueType::RGB:
        for ( auto i = 0u; i < size; ++i )
            dst[i] = scalarToRgb( src + i * 3, 3 );
        break;
    case TiffParameters::ValueType::Scalar:
        for ( auto i = 0u; i < size; ++i )
            dst[i] = scalarToRgb( src[i] );
        break;
    case TiffParameters::ValueType::Unknown:
        MR_UNREACHABLE_NO_RETURN;
        break;
    }
}

template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void copySampledData( T* dst, const U* src, size_t size, TiffParameters::ValueType valueType )
{
    switch ( valueType )
    {
    case TiffParameters::ValueType::Scalar:
        std::copy_n( src, size, dst );
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

using ReadTiffCallback = std::function<void ( const std::byte* data, size_t sizeInBytes, size_t offsetX, size_t offsetY )>;

void readTiff( TIFF* tiff, const TiffParameters& tp, ReadTiffCallback cb )
{
    const auto pixelSize = tp.getPixelSize();
    assert( pixelSize != 0 );

    Buffer<std::byte> buffer;
    if ( !tp.tileSize )
    {
        buffer.resize( (size_t)tp.imageSize.x * pixelSize );

        for ( auto row = 0; row < tp.imageSize.y; ++row )
        {
            TIFFReadScanline( tiff, buffer.data(), (uint32_t)row );

            cb( buffer.data(), buffer.size(), 0, row );
        }
    }
    else
    {
        buffer.resize( (size_t)tp.tileSize->x * tp.tileSize->y * pixelSize );
        const auto tileRowSize = (size_t)tp.tileSize->x * pixelSize;

        for ( auto tileY : splitByChunks( tp.imageSize.y, tp.tileSize->y ) )
        {
            for ( auto tileX : splitByChunks( tp.imageSize.x, tp.tileSize->x ) )
            {
                TIFFReadTile( tiff, buffer.data(), (uint32_t)tileX.offset, (uint32_t)tileY.offset, 0, 0 );

                for ( auto tileRow = 0u; tileRow < tileY.size; ++tileRow )
                    cb( buffer.data() + tileRow * tileRowSize, tileX.size * pixelSize, tileX.offset, tileY.offset + tileRow );
            }
        }
    }
}

void readTiff( TIFF* tiff, const TiffParameters& tp, std::byte* data )
{
    return readTiff( tiff, tp, [&] ( const std::byte* bytes, size_t size, size_t offsetX, size_t offsetY )
    {
        std::copy_n( bytes, size, data + offsetY * tp.imageSize.x + offsetX );
    } );
}

} // namespace

namespace MR
{

namespace DistanceMapLoad
{

Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld* dmapToWorld )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    auto params = readTiffParameters( tiff );

    if ( dmapToWorld )
        if ( const auto xf = readGeoTiffParameters( tiff ) )
            *dmapToWorld = *xf;

    DistanceMap result { (size_t)params.imageSize.x, (size_t)params.imageSize.y };

    readTiff( tiff, params, [&] ( const std::byte* bytes, size_t size, size_t offsetX, size_t offsetY )
    {
        visitData( bytes, params.getDataType(), [&] <typename T> ( const T* data )
        {
            copySampledData( result.data() + offsetY * result.resX() + offsetX, data, size / params.getPixelSize(), params.valueType );
        } );
    } );

    return result;
}

} // namespace DistanceMapLoad

namespace DistanceMapSave
{

Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, DistanceMapToWorld* dmapToWorld )
{
    auto tiff = TIFFOpen( utf8string( path ).c_str(), "w" );
    if ( !tiff )
        return unexpected( "Cannot write file: " + utf8string( path ) );
    MR_FINALLY {
        TIFFClose( tiff );
    };

    TIFFSetField( tiff, TIFFTAG_IMAGEWIDTH, dmap.resX() );
    TIFFSetField( tiff, TIFFTAG_IMAGELENGTH, dmap.resY() );

    // 32-bit float
    TIFFSetField( tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISWHITE );
    TIFFSetField( tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP );
    TIFFSetField( tiff, TIFFTAG_BITSPERSAMPLE, 32 );
    TIFFSetField( tiff, TIFFTAG_SAMPLESPERPIXEL, 1 );

    // FIXME: orientation is ignored
    TIFFSetField( tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );
    TIFFSetField( tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );

    if ( dmapToWorld )
    {
        const Matrix4d matrix { AffineXf3d( dmapToWorld->xf() ) };
        TIFFSetField( tiff, GeoTiff::ModelTransformationTag, 16, (double*)&matrix );
    }

    for ( auto row = 0; row < dmap.resY(); ++row )
    {
        // FIXME: orientation is ignored
        const auto* data = dmap.data() + ( dmap.resY() - 1 - row ) * dmap.resX();
        TIFFWriteScanline( tiff, (void*)data, row );
    }
    TIFFFlush( tiff );

    return {};
}

} // namespace DistanceMapSave

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
        readTiff( tiff, params, [&] ( const std::byte* bytes, size_t size, size_t offsetX, size_t offsetY )
        {
            visitData( bytes, params.getDataType(), [&] <typename T> ( const T* data )
            {
                copySampledData( result.pixels.data() + offsetY * params.imageSize.x + offsetX, data, size / sizeof( T ), params.valueType );
            } );
        } );
    }
    else
    {
        assert( params.valueType == TiffParameters::ValueType::Scalar );

        Buffer<std::byte> buffer;
        buffer.resize( result.pixels.size() * params.getPixelSize() );

        readTiff( tiff, params, buffer.data() );

        visitData( buffer.data(), params.getDataType(), [&] <typename U> ( const U* data )
        {
            const auto [min, max] = parallelMinMax( data, result.pixels.size() );
            ParallelFor( (size_t)0, result.pixels.size(), [&, min = min, max = max] ( size_t i )
            {
                result.pixels[i] = scalarToRgb( data[i], min, max );
            } );
        } );
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

    for ( auto row = 0; row < image.resolution.y; ++row )
    {
        // FIXME: orientation is ignored
        const auto* data = image.pixels.data() + (size_t)( image.resolution.y - 1 - row ) * image.resolution.x;
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
