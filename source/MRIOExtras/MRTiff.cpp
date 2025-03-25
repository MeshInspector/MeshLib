#include "MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF

#include "MRMesh/MRBitSetParallelFor.h"
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

template <typename... Args>
constexpr auto makeVariantArray( std::variant<Args...> )
{
    return std::array<std::variant<Args...>, sizeof...( Args )> { Args{}... };
}

template <typename Variant>
constexpr auto makeVariantArray()
{
    return makeVariantArray( Variant{} );
}

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

using DataVariant = std::variant<
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    float,
    double
>;

using DataPtrVariant = std::variant<
    uint8_t*,
    uint16_t*,
    uint32_t*,
    uint64_t*,
    int8_t*,
    int16_t*,
    int32_t*,
    int64_t*,
    float*,
    double*
>;

template <typename T>
DataType from()
{
    static constexpr auto variant = DataVariant( T() );
    return DataType( variant.index() );
}

template <typename Visitor>
auto visit( Visitor&& vis, DataType type )
{
    static constexpr auto variants = makeVariantArray<DataVariant>();
    return std::visit( vis, variants.at( (size_t)type ) );
}

template <typename Visitor>
auto visit( const std::byte* data, size_t size, Visitor&& vis, DataType type )
{
    return visit( [=] <typename T> ( T )
    {
        return vis( reinterpret_cast<const T*>( data ), size / sizeof( T ) );
    }, type );
}

template <typename Visitor>
auto visit( std::byte* data, size_t size, Visitor&& vis, DataType type )
{
    return visit( [=] <typename T> ( T )
    {
        return vis( reinterpret_cast<T*>( data ), size / sizeof( T ) );
    }, type );
}

enum class SampleType
{
    Scalar,
    RGB,
    RGBA,
};

size_t sizeOf( SampleType samples )
{
    using enum SampleType;
    switch ( samples )
    {
    case Scalar:
        return 1;
    case RGB:
        return 3;
    case RGBA:
        return 4;
    }
    MR_UNREACHABLE
}

template <typename T, typename Func>
void iterateSamples( SampleType samples, T* data, size_t size, Func&& f )
{
    const auto step = sizeOf( samples );
    for ( auto i = 0u; i < size / step; ++i )
        f( i, data + i * step );
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

    // optional NoData value
    std::optional<std::string> noDataValue;

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

    [[nodiscard]] ::SampleType getSampleType() const
    {
        switch ( valueType )
        {
        case ValueType::Scalar:
            return ::SampleType::Scalar;
        case ValueType::RGB:
            return ::SampleType::RGB;
        case ValueType::RGBA:
            return ::SampleType::RGBA;
        case ValueType::Unknown:
            MR_UNREACHABLE
        }
    }

    [[nodiscard]] std::optional<DataVariant> getNoDataValue() const
    {
        if ( !noDataValue )
            return std::nullopt;

        const auto& s = *noDataValue;
        return visit( [&] <typename T> ( T value ) -> std::optional<DataVariant>
        {
            if ( std::from_chars( s.data(), s.data() + s.size(), value ).ec != std::errc{} )
                return std::nullopt;

            return DataVariant( value );
        }, getDataType() );
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

    char *gdalNoData {};
    uint32_t count {};
    if ( TIFFGetField( tiff, TIFFTAG_GDAL_NODATA, &count, &gdalNoData ) && count != 0 )
        params.noDataValue.emplace( gdalNoData );

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

Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld* dmapToWorld, ProgressCallback )
{
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    auto params = readTiffParameters( tiff );

    if ( dmapToWorld )
        if ( const auto xf = readGeoTiffParameters( tiff ) )
            *dmapToWorld = *xf;

    DistanceMap result { (size_t)params.imageSize.x, (size_t)params.imageSize.y };

    readTiff( tiff, params, [&] ( const std::byte* bytes, size_t bytesSize, size_t offsetX, size_t offsetY )
    {
        visit( bytes, bytesSize, [&] <typename T> ( const T* data, size_t size )
        {
            const T* noDataValue = nullptr;
            if ( const auto variant = params.getNoDataValue() )
                noDataValue = std::get_if<T>( &(*variant) );

            iterateSamples( params.getSampleType(), data, size, [&] ( size_t i, const T* sample )
            {
                if ( noDataValue && sample[0] == *noDataValue )
                    return;

                float value;
                switch ( params.getSampleType() )
                {
                case SampleType::Scalar:
                    value = sample[0];
                    break;
                case SampleType::RGB:
                case SampleType::RGBA:
                    // luma/brightness component from the YCbCr color space
                    value =
                          (float)sample[0] * 0.299f
                        + (float)sample[1] * 0.587f
                        + (float)sample[2] * 0.114f
                    ;
                }
                result.set( offsetX + i, offsetY, value );
            } );
        }, params.getDataType() );
    } );

    return result;
}

MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "GeoTIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff )

} // namespace DistanceMapLoad

namespace DistanceMapSave
{

Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const AffineXf3f* xf )
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

    if ( xf )
    {
        const Matrix4d matrix { AffineXf3d( *xf ) };
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

// FIXME: single filter
MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "GeoTIFF (.tif)", "*.tif" ), toTiff )
MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "GeoTIFF (.tiff)", "*.tiff" ), toTiff )

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
        readTiff( tiff, params, [&] ( const std::byte* bytes, size_t bytesSize, size_t offsetX, size_t offsetY )
        {
            visit( bytes, bytesSize, [&] <typename T> ( const T* data, size_t size )
            {
                const auto offset = offsetY * params.imageSize.x + offsetX;
                iterateSamples( params.getSampleType(), data, size, [&] ( size_t i, const T* sample )
                {
                    result.pixels[offset + i] = Color {
                        Color::valToUint8( sample[0] ),
                        Color::valToUint8( sample[1] ),
                        Color::valToUint8( sample[2] ),
                        params.getSampleType() == SampleType::RGBA ? Color::valToUint8( sample[3] ) : 255,
                    };
                } );
            }, params.getDataType() );
        } );
    }
    else
    {
        assert( params.valueType == TiffParameters::ValueType::Scalar );

        Buffer<std::byte> buffer;
        buffer.resize( result.pixels.size() * params.getPixelSize() );

        readTiff( tiff, params, buffer.data() );

        visit( buffer.data(), buffer.size(), [&] <typename T> ( const T* data, size_t size )
        {
            assert( size == result.pixels.size() );

            const T* noDataValue = nullptr;
            if ( const auto variant = params.getNoDataValue() )
                noDataValue = std::get_if<T>( &(*variant) );

            BitSet valid;
            if ( noDataValue )
            {
                valid.resize( result.pixels.size() );
                BitSetParallelForAll( valid, [&] ( size_t i )
                {
                    valid.set( i, data[i] != *noDataValue );
                } );
            }

            const auto [min, max] = parallelMinMax( data, result.pixels.size(), valid.count() != 0 ? &valid : nullptr );
            ParallelFor( (size_t)0, result.pixels.size(), [&, min = min, max = max] ( size_t i )
            {
                if ( noDataValue && data[i] == *noDataValue )
                    return;

                const auto value = (float)( data[i] - min ) / (float)( max - min );
                result.pixels[i] = Color {
                    Color::valToUint8( value ),
                    Color::valToUint8( value ),
                    Color::valToUint8( value ),
                };
            } );
        }, params.getDataType() );
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
