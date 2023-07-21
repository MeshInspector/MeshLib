#ifndef MRMESH_NO_TIFF
#include "MRReadTIF.h"
#include "MRStringConvert.h"
#include "MRBuffer.h"

#include <tiffio.h>

namespace MR
{

template<typename SampleType>
void setDataValue( float* data, const SampleType* input, TifParameters::ValueType type,
    float* min, float* max )
{
    float res = 0;

    switch ( type )
    {
    case MR::TifParameters::ValueType::Scalar:
        res = float( *input );
        break;
    case MR::TifParameters::ValueType::RGB:
    case MR::TifParameters::ValueType::RGBA:
        res =
            ( 0.299f * float( input[0] ) +
                0.587f * float( input[1] ) +
                0.114f * float( input[2] ) );
        break;
    case MR::TifParameters::ValueType::Unknown:
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
void readRawTif( TIFF* tif, float* data, const TifParameters& tp, float* min = nullptr, float* max = nullptr )
{
    assert( sizeof( SampleType ) == tp.bytesPerSample );
    int samplePerPixel = 0;
    if ( tp.valueType == TifParameters::ValueType::Scalar )
        samplePerPixel = 1;
    else if ( tp.valueType == TifParameters::ValueType::RGB )
        samplePerPixel = 3;
    else if ( tp.valueType == TifParameters::ValueType::RGBA )
        samplePerPixel = 4;
    assert( samplePerPixel != 0 );

    Buffer<SampleType> buffer( tp.tiled ? 
        ( size_t( tp.tileSize.x ) * tp.tileSize.y * samplePerPixel ) :
        ( size_t( tp.imageSize.x ) * samplePerPixel ) );

    if ( tp.tiled )
    {
        for ( int y = 0; y < tp.imageSize.y; y += tp.tileSize.y )
        {
            for ( int x = 0; x < tp.imageSize.x; x += tp.tileSize.x )
            {
                TIFFReadTile( tif, ( void* )( buffer.data() ), x, y, 0, 0 );
                for ( int y0 = y; y0 < std::min( y + tp.tileSize.y, tp.imageSize.y ); ++y0 )
                {
                    for ( int x0 = x; x0 < std::min( x + tp.tileSize.x, tp.imageSize.x ); ++x0 )
                    {
                        setDataValue( data + tp.imageSize.x * y0 + x0,
                            buffer.data() + samplePerPixel * ( tp.tileSize.x * ( y0 - y ) + ( x0 - x ) ), tp.valueType,
                            min, max );
                    }
                }
            }
        }
    }
    else
    {
        for ( uint32_t i = 0; i < uint32_t( tp.imageSize.y ); ++i )
        {
            TIFFReadScanline( tif, ( void* )( buffer.data() ), i );
            auto shift = i * tp.imageSize.x;
            for ( int j = 0; j < tp.imageSize.x; ++j )
                setDataValue( data + shift + j, buffer.data() + samplePerPixel * j, tp.valueType,
                    min, max );
        }
    }
}



class TifHolder 
{
public:
    TifHolder( const std::filesystem::path& path, const char* mode )
    {
        tifPtr_ = TIFFOpen( utf8string( path ).c_str(), mode );
    }
    ~TifHolder()
    {
        if ( !tifPtr_ )
            return;
        TIFFClose( tifPtr_ );
        tifPtr_ = nullptr;
    }
    operator TIFF* ( ) { return tifPtr_; }
    operator const TIFF* ( ) const { return tifPtr_; }
    operator bool() const { return bool( tifPtr_ ); }
private:
    TIFF* tifPtr_{ nullptr };
};

bool isTIFFile( const std::filesystem::path& path )
{
    TifHolder tif( path, "rh" );// only header
    return bool( tif );
}

Expected<TifParameters, std::string> readTifParameters( TIFF* tif )
{
    TifParameters params;

    int bitsPerSample = 0;
    TIFFGetField( tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample );
    params.bytesPerSample = bitsPerSample >> 3; // convert to bytes

    int samplePerPixel = 0;
    TIFFGetField( tif, TIFFTAG_SAMPLESPERPIXEL, &samplePerPixel );
    if ( samplePerPixel == 1 )
        params.valueType = TifParameters::ValueType::Scalar;
    else if ( samplePerPixel == 3 )
        params.valueType = TifParameters::ValueType::RGB;
    else if ( samplePerPixel == 4 )
        params.valueType = TifParameters::ValueType::RGBA;

    int sampleFormat = 0;
    TIFFGetField( tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat );
    if ( sampleFormat == SAMPLEFORMAT_UINT || sampleFormat == 0 )
        params.sampleType = TifParameters::SampleType::Uint;
    else if ( sampleFormat == SAMPLEFORMAT_INT )
        params.sampleType = TifParameters::SampleType::Int;
    else if ( sampleFormat == SAMPLEFORMAT_IEEEFP )
        params.sampleType = TifParameters::SampleType::Float;

    TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &params.imageSize.x );
    TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &params.imageSize.y );

    params.tiled = bool( TIFFIsTiled( tif ) );
    if ( params.tiled )
    {
        TIFFGetField( tif, TIFFTAG_TILEWIDTH, &params.tileSize.x );
        TIFFGetField( tif, TIFFTAG_TILELENGTH, &params.tileSize.y );

        TIFFGetField( tif, TIFFTAG_TILEDEPTH, &params.depth );
        if ( params.depth != 0 )
            params.layers = int( TIFFNumberOfTiles( tif ) );
    }

    if ( params.valueType == TifParameters::ValueType::Unknown ||
        params.sampleType == TifParameters::SampleType::Unknown )
        return unexpected( "Unsupported pixel format" );

    if ( params.depth != 0 )
        return unexpected( "Unsupported tiles format" );

    return params;
}

Expected<TifParameters, std::string> readTifParameters( const std::filesystem::path& path )
{
    TifHolder tif( path, "r" );
    if ( !tif )
        return unexpected( "Cannot read file: " + utf8string( path ) );

    return addFileNameInError( readTifParameters( tif ), path );
}

VoidOrErrStr readRawTif( const std::filesystem::path& path, RawTifOutput& output )
{
    TifHolder tif( path, "r" );
    if ( !tif )
        return unexpected( "Cannot read file: " + utf8string( path ) );
    auto localParams = readTifParameters( tif );
    if ( !localParams.has_value() )
        return unexpected( localParams.error() + ": " + utf8string( path ) );
    if ( output.params && *output.params != *localParams )
        return unexpected( "Inconsistent parameters in file: " + utf8string( path ) );

    if ( localParams->sampleType == TifParameters::SampleType::Uint )
    {
        if ( localParams->bytesPerSample == sizeof( uint8_t ) )
            readRawTif<uint8_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint16_t ) )
            readRawTif<uint16_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint32_t ) )
            readRawTif<uint32_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( uint64_t ) )
            readRawTif<uint64_t>( tif, output.data, *localParams, output.min, output.max );
    }
    else if ( localParams->sampleType == TifParameters::SampleType::Int )
    {
        if ( localParams->bytesPerSample == sizeof( int8_t ) )
            readRawTif<int8_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int16_t ) )
            readRawTif<int16_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int32_t ) )
            readRawTif<int32_t>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( int64_t ) )
            readRawTif<int64_t>( tif, output.data, *localParams, output.min, output.max );
    }
    else if ( localParams->sampleType == TifParameters::SampleType::Float )
    {
        if ( localParams->bytesPerSample == sizeof( float ) )
            readRawTif<float>( tif, output.data, *localParams, output.min, output.max );
        else if ( localParams->bytesPerSample == sizeof( double ) )
            readRawTif<double>( tif, output.data, *localParams, output.min, output.max );
    }
    return {};
}

}

#endif