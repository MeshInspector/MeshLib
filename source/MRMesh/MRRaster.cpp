#include "MRRaster.h"
#include "MRBox.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

#include <cmath>
#include <cstring>

namespace MR
{

size_t RasterInfo::dataSize() const
{
    if ( dims.x <= 0 || dims.y <= 0 || dims.z <= 0 )
        return 0;
    return size_t( dims.x ) * size_t( dims.y ) * size_t( dims.z ) * getScalarTypeSize( type );
}

Expected<Image> convertRasterToImage( const Raster& raster )
{
    MR_TIMER;
    const auto& info = raster.info;
    const auto valueSize = getScalarTypeSize( info.type );
    if ( valueSize == 0 )
        return unexpected( "Unsupported value type" );
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
    const auto* data = raster.data.data();

    // fills the image with the colors of the values
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

    switch ( info.type )
    {
    case ScalarType::RGBA8:
        fill( [data] ( size_t i )
        {
            Color c;
            std::memcpy( (void*)&c, data + i * sizeof( Color ), sizeof( Color ) );
            return c;
        } );
        break;
    case ScalarType::RGB8:
        fill( [data] ( size_t i )
        {
            const auto* rgb = data + i * 3;
            return Color( rgb[0], rgb[1], rgb[2] );
        } );
        break;
    case ScalarType::UInt8:
        fill( [data] ( size_t i )
        {
            return Color( data[i], data[i], data[i] );
        } );
        break;
    case ScalarType::UInt16:
        fill( [data] ( size_t i )
        {
            uint16_t v;
            std::memcpy( &v, data + i * sizeof( v ), sizeof( v ) );
            // the high byte, as in libtiff's RGBA reader
            const auto g = v >> 8;
            return Color( g, g, g );
        } );
        break;
    default:
    {
        auto getValue = [data, valueSize, type = info.type] ( size_t i )
        {
            return visitScalarType( [] ( auto v ) { return double( v ); }, type, ( const char* )data + i * valueSize );
        };
        const auto valueRange = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, width * height ), Box1d{},
            [&] ( const tbb::blocked_range<size_t>& range, Box1d box )
            {
                for ( auto i = range.begin(); i < range.end(); ++i )
                {
                    const auto v = getValue( i );
                    if ( !std::isnan( v ) )
                        box.include( v );
                }
                return box;
            },
            [] ( Box1d a, const Box1d& b )
            {
                a.include( b );
                return a;
            } );
        const auto rangeSize = valueRange.valid() ? valueRange.max - valueRange.min : 0.;
        fill( [&] ( size_t i )
        {
            const auto v = getValue( i );
            if ( std::isnan( v ) )
                return Color::black();
            const auto g = rangeSize > 0 ? uint8_t( 255. * ( v - valueRange.min ) / rangeSize ) : uint8_t( 0 );
            return Color( g, g, g );
        } );
        break;
    }
    }

    return res;
}

Raster convertImageToRaster( const Image& image )
{
    MR_TIMER;
    Raster res{
        .info = {
            .dims = Vector3i( image.resolution.x, image.resolution.y, 1 ),
            .type = ScalarType::RGBA8,
        },
    };
    res.data.resize( res.info.dataSize() );
    assert( res.data.size() == image.pixels.size() * sizeof( Color ) );

    const auto width = size_t( image.resolution.x );
    const auto height = size_t( image.resolution.y );
    ParallelFor( size_t( 0 ), height, [&] ( size_t y )
    {
        // Image starts from the bottom row
        std::memcpy( res.data.data() + y * width * sizeof( Color ), (const void*)( image.pixels.data() + ( height - 1 - y ) * width ),
            width * sizeof( Color ) );
    } );
    return res;
}

} // namespace MR
