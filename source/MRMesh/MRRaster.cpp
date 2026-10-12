#include "MRRaster.h"
#include "MRBox.h"
#include "MRImage.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace MR
{

namespace
{

// converts a color component to 8 bits: integer values keep their high byte and are taken as unsigned ones, as in libtiff's RGBA reader;
// floating-point values are clamped to [0, 1], NaN becomes 0
template <typename T>
uint8_t colorTo8Bit( T v )
{
    if constexpr ( std::is_floating_point_v<T> )
        return std::isnan( v ) ? uint8_t( 0 ) : Color::valToUint8( v );
    else
        return uint8_t( std::make_unsigned_t<T>( v ) >> ( 8 * sizeof( T ) - 8 ) );
}

} // namespace

size_t RasterInfo::dataSize() const
{
    if ( dims.x <= 0 || dims.y <= 0 || dims.z <= 0 || channels <= 0 )
        return 0;
    // the product of three 31-bit dimensions, the number of channels and the value size can exceed 64 bits
    size_t res = getScalarTypeSize( type );
    for ( int d : { dims.x, dims.y, dims.z, channels } )
    {
        if ( res > std::numeric_limits<size_t>::max() / size_t( d ) )
            return std::numeric_limits<size_t>::max();
        res *= size_t( d );
    }
    return res;
}

Expected<Image> convertRasterToImage( const Raster& raster )
{
    MR_TIMER;
    const auto& info = raster.info;
    const auto valueSize = getScalarTypeSize( info.type );
    if ( valueSize == 0 )
        return unexpected( "Unsupported value type" );
    if ( info.channels != 1 && info.channels < 3 )
        return unexpected( "Unsupported number of channels" );
    if ( info.dims.z != 1 )
        return unexpected( "Only a raster with one layer can be converted to an image" );
    if ( info.dims.x < 0 || info.dims.y < 0 || raster.data.size() != info.dataSize() )
        return unexpected( "Raster data size does not match its dimensions" );

    const auto width = size_t( info.dims.x );
    const auto height = size_t( info.dims.y );
    Image res{
        .resolution = Vector2i( info.dims.x, info.dims.y ),
    };
    res.pixels.resize( width * height );
    const auto* data = ( const char* )raster.data.data();
    const auto pixelSize = valueSize * size_t( info.channels );

    // fills the image with the colors of the pixels
    auto fill = [&] ( auto&& getColor )
    {
        ParallelFor( size_t( 0 ), height, [&] ( size_t y )
        {
            // Image starts from the bottom row
            auto* dst = res.pixels.data() + ( height - 1 - y ) * width;
            for ( size_t x = 0; x < width; ++x )
                dst[x] = getColor( y * width + x );
        } );
    };

    if ( info.channels >= 3 )
    {
        const auto components = std::min( info.channels, 4 );
        fill( [&] ( size_t i )
        {
            uint8_t c[4] = { 0, 0, 0, 255 };
            for ( int j = 0; j < components; ++j )
                c[j] = visitScalarType( [] ( auto v ) { return colorTo8Bit( v ); }, info.type, data + i * pixelSize + j * valueSize );
            return Color( c[0], c[1], c[2], c[3] );
        } );
        return res;
    }

    auto getValue = [&] ( size_t i )
    {
        return visitScalarType( [] ( auto v ) { return double( v ); }, info.type, data + i * pixelSize );
    };
    const auto valueRange = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, width * height ), Box1d{},
        [&] ( const tbb::blocked_range<size_t>& range, Box1d box )
        {
            for ( auto i = range.begin(); i < range.end(); ++i )
            {
                const auto v = getValue( i );
                if ( std::isfinite( v ) )
                    box.include( v );
            }
            return box;
        },
        [] ( Box1d a, const Box1d& b )
        {
            a.include( b );
            return a;
        } );
    // the halves do not overflow even for the range of all double values
    const auto halfMin = valueRange.min / 2;
    const auto halfRangeSize = valueRange.valid() ? valueRange.max / 2 - halfMin : 0.;
    fill( [&] ( size_t i )
    {
        const auto v = getValue( i );
        if ( !std::isfinite( v ) )
            return Color::black();
        const auto t = halfRangeSize > 0 ? std::clamp( ( v / 2 - halfMin ) / halfRangeSize, 0., 1. ) : 0.;
        const auto g = uint8_t( 255. * t );
        return Color( g, g, g );
    } );
    return res;
}

Expected<void> convertRasterToFloats( const Raster& raster, std::vector<float>& dst )
{
    MR_TIMER;
    const auto& info = raster.info;
    const auto valueSize = getScalarTypeSize( info.type );
    if ( valueSize == 0 )
        return unexpected( "Unsupported value type" );
    if ( info.channels != 1 && info.channels < 3 )
        return unexpected( "Unsupported number of channels" );
    if ( raster.data.size() != info.dataSize() )
        return unexpected( "Raster data size does not match its dimensions" );

    const auto channels = size_t( info.channels );
    dst.resize( raster.data.size() / ( valueSize * channels ) );
    const auto* data = ( const char* )raster.data.data();
    auto getValue = [&] ( size_t i )
    {
        return visitScalarType( [] ( auto v ) { return float( v ); }, info.type, data + i * valueSize );
    };
    ParallelFor( size_t( 0 ), dst.size(), [&] ( size_t i )
    {
        const auto j = i * channels;
        dst[i] = channels == 1 ? getValue( j ) : 0.299f * getValue( j ) + 0.587f * getValue( j + 1 ) + 0.114f * getValue( j + 2 );
    } );
    return {};
}

} // namespace MR
