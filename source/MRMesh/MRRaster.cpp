#include "MRRaster.h"
#include "MRBox.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

#include <cmath>
#include <cstring>

namespace MR
{

namespace
{

// calls f( T{} ) with T being the type of the samples; sampleType must be supported in rasters
template <typename F>
decltype( auto ) visitSampleType( ScalarType sampleType, F&& f )
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
    case ScalarType::Float32_4:
    case ScalarType::Unknown:
    case ScalarType::Count:
        break;
    }
    MR_UNREACHABLE
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

} // namespace

size_t RasterInfo::sampleSize() const
{
    switch ( sampleType )
    {
    case ScalarType::UInt8:
    case ScalarType::Int8:
        return 1;
    case ScalarType::UInt16:
    case ScalarType::Int16:
        return 2;
    case ScalarType::UInt32:
    case ScalarType::Int32:
    case ScalarType::Float32:
        return 4;
    case ScalarType::UInt64:
    case ScalarType::Int64:
    case ScalarType::Float64:
        return 8;
    case ScalarType::Float32_4:
    case ScalarType::Unknown:
    case ScalarType::Count:
        break;
    }
    return 0;
}

size_t RasterInfo::dataSize() const
{
    if ( resolution.x <= 0 || resolution.y <= 0 || channels <= 0 )
        return 0;
    return size_t( resolution.x ) * size_t( resolution.y ) * size_t( channels ) * sampleSize();
}

Expected<Image> convertRasterToImage( const Raster& raster )
{
    MR_TIMER;
    const auto& info = raster.info;
    if ( info.sampleSize() == 0 )
        return unexpected( "Unsupported sample type" );
    if ( info.resolution.x < 0 || info.resolution.y < 0 || info.channels <= 0 || raster.data.size() != info.dataSize() )
        return unexpected( "Raster data size does not match its resolution" );

    const auto width = size_t( info.resolution.x );
    const auto height = size_t( info.resolution.y );
    const auto channels = size_t( info.channels );
    Image res{
        .resolution = info.resolution,
    };
    res.pixels.resize( width * height );

    visitSampleType( info.sampleType, [&] <typename T> ( T )
    {
        const auto* data = raster.data.data();
        auto sample = [data] ( size_t i ) { return getSample<T>( data, i ); };

        const bool usePalette = std::is_integral_v<T> && channels == 1 && !info.palette.empty();
        const bool isGray = channels <= 2 && !usePalette;

        // gray values of other types than 8-bit and 16-bit unsigned are scaled from the range of valid values
        constexpr bool scaleGray = !std::is_same_v<T, uint8_t> && !std::is_same_v<T, uint16_t>;
        auto isValidGray = [&info] ( T v )
        {
            if constexpr ( std::is_floating_point_v<T> )
            {
                if ( std::isnan( v ) )
                    return false;
            }
            return !info.noData || double( v ) != *info.noData;
        };
        Box1d validRange;
        if ( scaleGray && isGray )
        {
            validRange = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, width * height ), Box1d{},
                [&] ( const tbb::blocked_range<size_t>& range, Box1d box )
                {
                    for ( auto i = range.begin(); i < range.end(); ++i )
                    {
                        const auto v = sample( i * channels );
                        if ( isValidGray( v ) )
                            box.include( double( v ) );
                    }
                    return box;
                },
                [] ( Box1d a, const Box1d& b )
                {
                    a.include( b );
                    return a;
                } );
        }
        [[maybe_unused]] const auto grayRange = validRange.valid() ? validRange.max - validRange.min : 0.;

        auto grayColor = [&] ( T v ) -> Color
        {
            uint8_t g = 0;
            if constexpr ( std::is_same_v<T, uint8_t> )
            {
                g = v;
            }
            else if constexpr ( std::is_same_v<T, uint16_t> )
            {
                // the high byte, as in libtiff's RGBA reader
                g = uint8_t( v >> 8 );
            }
            else
            {
                if ( !isValidGray( v ) )
                    return Color::black();
                if ( grayRange > 0 )
                    g = uint8_t( 255. * ( double( v ) - validRange.min ) / grayRange );
            }
            if ( info.minIsWhite )
                g = uint8_t( 255 - g );
            return Color( g, g, g );
        };

        auto paletteColor = [&] ( [[maybe_unused]] T v ) -> Color
        {
            if constexpr ( std::is_integral_v<T> )
            {
                if constexpr ( std::is_signed_v<T> )
                {
                    if ( v < 0 )
                        return Color::black();
                }
                if ( uint64_t( v ) < info.palette.size() )
                    return info.palette[size_t( v )];
            }
            return Color::black();
        };

        ParallelFor( size_t( 0 ), height, [&] ( size_t y )
        {
            // Image starts from the bottom row
            auto* dst = res.pixels.data() + ( height - 1 - y ) * width;
            for ( size_t x = 0; x < width; ++x )
            {
                const auto i = ( y * width + x ) * channels;
                Color c;
                if ( usePalette )
                {
                    c = paletteColor( sample( i ) );
                }
                else if ( isGray )
                {
                    c = grayColor( sample( i ) );
                    if ( channels == 2 )
                        c.a = colorTo8Bit( sample( i + 1 ) );
                }
                else
                {
                    c = Color( colorTo8Bit( sample( i ) ), colorTo8Bit( sample( i + 1 ) ), colorTo8Bit( sample( i + 2 ) ),
                        channels >= 4 ? colorTo8Bit( sample( i + 3 ) ) : uint8_t( 255 ) );
                }
                dst[x] = c;
            }
        } );
    } );

    return res;
}

Raster convertImageToRaster( const Image& image )
{
    MR_TIMER;
    Raster res{
        .info = {
            .resolution = image.resolution,
            .channels = 4,
            .sampleType = ScalarType::UInt8,
        },
    };
    res.data.resize( res.info.dataSize() );
    assert( res.data.size() == image.pixels.size() * sizeof( Color ) );

    const auto width = size_t( image.resolution.x );
    const auto height = size_t( image.resolution.y );
    ParallelFor( size_t( 0 ), height, [&] ( size_t y )
    {
        // Image starts from the bottom row
        std::memcpy( res.data.data() + y * width * sizeof( Color ), image.pixels.data() + ( height - 1 - y ) * width, width * sizeof( Color ) );
    } );
    return res;
}

} // namespace MR
