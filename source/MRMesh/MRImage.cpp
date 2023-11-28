#include "MRImage.h"

namespace MR
{

Color Image::sampleDiscrete( const UVCoord & pos ) const
{
    const float x = std::clamp( pos.x, 0.0f, 1.0f ) * ( resolution.x - 1 );
    const float y = std::clamp( pos.y, 0.0f, 1.0f ) * ( resolution.y - 1 );
    return operator[]( { (int)std::lround( x ), (int)std::lround( y ) } );
}

Color Image::sampleBilinear( const UVCoord & pos ) const
{
    const float x = std::clamp( pos.x, 0.0f, 1.0f ) * ( resolution.x - 1 );
    const float y = std::clamp( pos.y, 0.0f, 1.0f ) * ( resolution.y - 1 );

    const float xlowf = std::floor( x );
    const float ylowf = std::floor( y );
    const int xlow = int( xlowf );
    const int ylow = int( ylowf );
    assert( 0 <= xlow && xlow < resolution.x );
    assert( 0 <= ylow && ylow < resolution.y );

    const auto idx = xlow + ylow * resolution.x;
    const auto lowlow =   pixels[idx];
    const auto lowhigh =  ( ylow + 1 < resolution.y ) ? pixels[idx + resolution.x] : Color();
    const auto highlow =  ( xlow + 1 < resolution.x ) ? pixels[idx + 1] : Color();
    const auto highhigh = ( ylow + 1 < resolution.y ) && ( xlow + 1 < resolution.x ) ? pixels[idx + resolution.x + 1] : Color();

    auto v = []( const Color & c )
        { return Vector4f( (float)c.r, (float)c.g, (float)c.b, (float)c.a ); };

    auto c = []( const Vector4f & v )
    {
        return Color(
            uint8_t( std::lround( v.x ) ),
            uint8_t( std::lround( v.y ) ),
            uint8_t( std::lround( v.z ) ),
            uint8_t( std::lround( v.w ) ) );
    };

    const float dx = x - xlowf;
    const float dy = y - ylowf;
    return c(
        ( v( lowlow ) * ( 1 - dy ) + v( lowhigh ) * dy ) * ( 1 - dx ) +
        ( v( highlow ) * ( 1 - dy ) + v( highhigh ) * dy ) * dx
    );
}

} //namespace MR
