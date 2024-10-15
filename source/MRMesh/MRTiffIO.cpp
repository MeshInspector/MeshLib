#include "MRTiffIO.h"
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
#include "MRStringConvert.h"
#include "MRBuffer.h"
#include "MRMatrix4.h"
#include "MRAffineXf3.h"

#include <tiffio.h>

namespace MR
{

template<typename SampleType>
void setDataValue( float* data, const SampleType* input, TiffParameters::ValueType type,
    float* min, float* max )
{
    float res = 0;

    switch ( type )
    {
    case MR::TiffParameters::ValueType::Scalar:
        res = float( *input );
        break;
    case MR::TiffParameters::ValueType::RGB:
    case MR::TiffParameters::ValueType::RGBA:
        res =
            ( 0.299f * float( input[0] ) +
                0.587f * float( input[1] ) +
                0.114f * float( input[2] ) );
        break;
    case MR::TiffParameters::ValueType::Unknown:
    default:
        break;
    }

    if ( min && res < *min )
        *min = res;
    if ( max && res > *max )
        *max = res;

    *data = res;
}

template<typename SampleType>
void readRawTiff( TIFF* tiff, uint8_t* bytes, size_t size, const TiffParameters& tp, 
    bool convertToFloat,
    float* min = nullptr, float* max = nullptr )
{
    assert( sizeof( SampleType ) == tp.bytesPerSample );
    int samplePerPixel = 0;
    if ( tp.valueType == TiffParameters::ValueType::Scalar )
        samplePerPixel = 1;
    else if ( tp.valueType == TiffParameters::ValueType::RGB )
        samplePerPixel = 3;
    else if ( tp.valueType == TiffParameters::ValueType::RGBA )
        samplePerPixel = 4;
    assert( samplePerPixel != 0 );

    Buffer<SampleType> buffer;
    if ( convertToFloat || tp.tiled )
    {
        buffer.resize( tp.tiled ?
            ( size_t( tp.tileSize.x ) * tp.tileSize.y * samplePerPixel ) :
            ( size_t( tp.imageSize.x ) * samplePerPixel ) );
    }

    if ( tp.tiled )
    {
        for ( int y = 0; y < tp.imageSize.y; y += tp.tileSize.y )
        {
            for ( int x = 0; x < tp.imageSize.x; x += tp.tileSize.x )
            {
                TIFFReadTile( tiff, ( void* )( buffer.data() ), x, y, 0, 0 );
                for ( int y0 = y; y0 < std::min( y + tp.tileSize.y, tp.imageSize.y ); ++y0 )
                {
                    size_t shift = tp.imageSize.x * y0;
                    if ( convertToFloat )
                    {
                        for ( int x0 = x; x0 < std::min( x + tp.tileSize.x, tp.imageSize.x ); ++x0 )
                        {
                            size_t dataShift = shift + x0;
                            if ( ( dataShift + 1 ) * sizeof( float ) > size )
                                continue;
                            setDataValue( ( float* )bytes + dataShift,
                                    buffer.data() + samplePerPixel * ( tp.tileSize.x * ( y0 - y ) + ( x0 - x ) ), tp.valueType,
                                    min, max );
                        }
                    }
                    else
                    {
                        size_t dataShift = shift + x;
                        auto modifier = samplePerPixel * tp.bytesPerSample;
                        if ( ( dataShift + tp.tileSize.x ) * modifier > size )
                            continue;
                        auto* first = ( const uint8_t* )( buffer.data() + samplePerPixel * ( tp.tileSize.x * ( y0 - y ) ) );
                        std::copy( first, first + modifier * tp.tileSize.x, bytes + dataShift * modifier );
                    }
                }
            }
        }
    }
    else
    {
        for ( uint32_t i = 0; i < uint32_t( tp.imageSize.y ); ++i )
        {
            auto shift = i * tp.imageSize.x;
            if ( !convertToFloat )
            {
                auto maxShift = shift + tp.imageSize.x;
                if ( maxShift * samplePerPixel * tp.bytesPerSample > size )
                    continue;
            }
            TIFFReadScanline( tiff, 
                convertToFloat ? 
                ( void* )( buffer.data() ) : 
                ( void* )( bytes + shift * samplePerPixel * tp.bytesPerSample ), i );
            if ( convertToFloat )
            {
                for ( int j = 0; j < tp.imageSize.x; ++j )
                {
                    size_t dataShift = shift + j;
                    if ( ( dataShift + 1 ) * sizeof( float ) > size )
                        continue;
                    setDataValue( ( float* )bytes + dataShift, buffer.data() + samplePerPixel * j, tp.valueType,
                        min, max );
                }
            }
        }
    }
}

Expected<void> writeRawTiff( const uint8_t* bytes, const std::filesystem::path& path, const BaseTiffParameters& params )
{
    TIFF* tif = TIFFOpen( MR::utf8string( path ).c_str(), "w" );
    if ( !tif )
        return unexpected("Cannot write file: "+ utf8string( path ) );

    TIFFSetField( tif, TIFFTAG_IMAGEWIDTH, params.imageSize.x );
    TIFFSetField( tif, TIFFTAG_IMAGELENGTH, params.imageSize.y );
    TIFFSetField( tif, TIFFTAG_BITSPERSAMPLE, params.bytesPerSample * 8 );
    int numSamples = 1;
    switch ( params.valueType )
    {
    case BaseTiffParameters::ValueType::Scalar:
        numSamples = 1;
        break;
    case BaseTiffParameters::ValueType::RGB:
        numSamples = 3;
        break;
    case BaseTiffParameters::ValueType::RGBA:
        numSamples = 4;
        break;
    default:
        break;
    }
    TIFFSetField( tif, TIFFTAG_SAMPLESPERPIXEL, numSamples );

    int sampleType = 0;
    switch ( params.sampleType )
    {
    case BaseTiffParameters::SampleType::Float:
        sampleType = SAMPLEFORMAT_IEEEFP;
        break;
    case BaseTiffParameters::SampleType::Uint:
        sampleType = SAMPLEFORMAT_UINT;
        break;
    case BaseTiffParameters::SampleType::Int:
        sampleType = SAMPLEFORMAT_INT;
        break;
    default:
        return unexpected( "Unknown sample format" );
        break;
    }

    TIFFSetField( tif, TIFFTAG_SAMPLEFORMAT, sampleType );

    TIFFSetField( tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISWHITE );

    for ( int row = 0; row < params.imageSize.y; row++ )
        TIFFWriteScanline( tif, ( void* )( bytes + row * params.imageSize.x * numSamples * params.bytesPerSample ), row );

    TIFFClose( tif );
    return {};
}

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

bool isTIFFFile( const std::filesystem::path& path )
{
    TiffHolder tiff( path, "rh" );// only header
    return bool( tiff );
}

Expected<TiffParameters> readTifParameters( TIFF* tiff )
{
    TiffParameters params;

    int bitsPerSample = 0;
    TIFFGetField( tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample );
    params.bytesPerSample = bitsPerSample >> 3; // convert to bytes

    int samplePerPixel = 0;
    TIFFGetField( tiff, TIFFTAG_SAMPLESPERPIXEL, &samplePerPixel );
    if ( samplePerPixel == 0 )
    {
        // incorrect tiff format, treat like Scalar
        samplePerPixel = 1;
    }
    if ( samplePerPixel == 1 )
        params.valueType = TiffParameters::ValueType::Scalar;
    else if ( samplePerPixel == 3 )
        params.valueType = TiffParameters::ValueType::RGB;
    else if ( samplePerPixel == 4 )
        params.valueType = TiffParameters::ValueType::RGBA;

    int sampleFormat = 0;
    TIFFGetField( tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    if ( sampleFormat == SAMPLEFORMAT_UINT || sampleFormat == 0 )
        params.sampleType = TiffParameters::SampleType::Uint;
    else if ( sampleFormat == SAMPLEFORMAT_INT )
        params.sampleType = TiffParameters::SampleType::Int;
    else if ( sampleFormat == SAMPLEFORMAT_IEEEFP )
        params.sampleType = TiffParameters::SampleType::Float;

    TIFFGetField( tiff, TIFFTAG_IMAGEWIDTH, &params.imageSize.x );
    TIFFGetField( tiff, TIFFTAG_IMAGELENGTH, &params.imageSize.y );

    params.tiled = bool( TIFFIsTiled( tiff ) );
    if ( params.tiled )
    {
        TIFFGetField( tiff, TIFFTAG_TILEWIDTH, &params.tileSize.x );
        TIFFGetField( tiff, TIFFTAG_TILELENGTH, &params.tileSize.y );

        TIFFGetField( tiff, TIFFTAG_TILEDEPTH, &params.depth );
        if ( params.depth != 0 )
            params.layers = int( TIFFNumberOfTiles( tiff ) );
    }

    if ( params.valueType == TiffParameters::ValueType::Unknown ||
        params.sampleType == TiffParameters::SampleType::Unknown )
        return unexpected( "Unsupported pixel format" );

    if ( params.depth != 0 )
        return unexpected( "Unsupported tiles format" );

    return params;
}

Expected<TiffParameters> readTiffParameters( const std::filesystem::path& path )
{
    TiffHolder tif( path, "r" );
    if ( !tif )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    return addFileNameInError( readTifParameters( tif ), path );
}

Expected<void> readRawTiff( const std::filesystem::path& path, RawTiffOutput& output )
{
    assert( output.size != 0 );
    if ( output.size == 0 )
        return unexpected( "Cannot read file to empty buffer" );
    TiffHolder tiff( path, "r" );
    if ( !tiff )
        return unexpected( "Cannot read file: " + utf8string( path ) );
    auto localParams = readTifParameters( tiff );
    if ( !localParams.has_value() )
        return unexpected( localParams.error() + ": " + utf8string( path ) );
    if ( output.params )
        *output.params = *localParams;

    if ( output.p2wXf )
    {
        // http://geotiff.maptools.org/spec/geotiff2.6.html
        constexpr uint32_t TIFFTAG_ModelTiePointTag = 33922;	/* GeoTIFF */
        constexpr uint32_t TIFFTAG_ModelPixelScaleTag = 33550;	/* GeoTIFF */
        constexpr uint32_t TIFFTAG_ModelTransformationTag = 34264;	/* GeoTIFF */
        Matrix4d matrix;
        if ( TIFFGetField( tiff, TIFFTAG_ModelTransformationTag, &matrix ) )
        {
            *output.p2wXf = AffineXf3f( Matrix4f( matrix ) );
        }
        else
        {
            double* dataTie;// will be freed with tiff
            uint32_t count;
            auto statusT = TIFFGetField( tiff, TIFFTAG_ModelTiePointTag, &count, &dataTie );
            if ( statusT && count == 6 )
            {
                Vector3d tiePoints[2];
                tiePoints[0] = { dataTie[0],dataTie[1],dataTie[2] };
                tiePoints[0] = { dataTie[3],dataTie[4],dataTie[5] };

                double* dataScale;// will be freed with tiff
                Vector3d scale;
                auto statusS = TIFFGetField( tiff, TIFFTAG_ModelPixelScaleTag, &count, &dataScale );
                if ( statusS && count == 3 )
                {
                    scale = { dataScale[0],dataScale[1],dataScale[2] };

                    output.p2wXf->A = Matrix3f::scale( float( scale.x ), -float( scale.y ),
                        scale.z == 0.0 ? 1.0f : float( scale.z ) );
                    output.p2wXf->b = Vector3f( tiePoints[1] );

                    output.p2wXf->b.x += float( tiePoints[0].x );
                    output.p2wXf->b.y += float( tiePoints[0].y );
                    if ( scale.z != 0.0 )
                        output.p2wXf->b.z += float( tiePoints[0].z );
                }
            }
        }
    }

    if ( localParams->sampleType == TiffParameters::SampleType::Uint )
    {
        if ( localParams->bytesPerSample == sizeof( uint8_t ) )
            readRawTiff<uint8_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint16_t ) )
            readRawTiff<uint16_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint32_t ) )
            readRawTiff<uint32_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint64_t ) )
            readRawTiff<uint64_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
    }
    else if ( localParams->sampleType == TiffParameters::SampleType::Int )
    {
        if ( localParams->bytesPerSample == sizeof( int8_t ) )
            readRawTiff<int8_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int16_t ) )
            readRawTiff<int16_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int32_t ) )
            readRawTiff<int32_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int64_t ) )
            readRawTiff<int64_t>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
    }
    else if ( localParams->sampleType == TiffParameters::SampleType::Float )
    {
        if ( localParams->bytesPerSample == sizeof( float ) )
            readRawTiff<float>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( double ) )
            readRawTiff<double>( tiff, output.bytes, output.size, *localParams, output.convertToFloat, output.min, output.max );
    }
    return {};
}

}

#endif