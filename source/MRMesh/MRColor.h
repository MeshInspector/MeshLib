#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRVector4.h"

namespace MR
{
struct Color
{
    uint8_t r, g, b, a;

    constexpr Color() noexcept : r{ 0 }, g{ 0 }, b{ 0 }, a{ 255 } {}
    explicit Color( NoInit ) noexcept {}
    constexpr Color( int r, int g, int b, int a ) noexcept : r{uint8_t(r)}, g{uint8_t(g)}, b{uint8_t(b)}, a{uint8_t(a)} {}
    constexpr Color( int r, int g, int b ) noexcept : r{uint8_t(r)}, g{uint8_t(g)}, b{uint8_t(b)}, a{255} {} // Using a separate overload instead of a default argument to produce better C bindings.
    constexpr Color( float r, float g, float b, float a ) noexcept :
        r{ r > 1 ? uint8_t( 255 ) : ( r < 0 ? uint8_t( 0 ) : uint8_t( r * 255 ) )},
        g{ g > 1 ? uint8_t( 255 ) : ( g < 0 ? uint8_t( 0 ) : uint8_t( g * 255 ) )},
        b{ b > 1 ? uint8_t( 255 ) : ( b < 0 ? uint8_t( 0 ) : uint8_t( b * 255 ) )},
        a{ a > 1 ? uint8_t( 255 ) : ( a < 0 ? uint8_t( 0 ) : uint8_t( a * 255 ) )} {}
    constexpr Color( float r, float g, float b ) noexcept : Color( r, g, b, 1.f ) {} // Using a separate overload instead of a default argument to produce better C bindings.

    constexpr unsigned int getUInt32() const noexcept { return ( unsigned( r ) ) + ( unsigned( g ) << 8 ) + ( unsigned( b ) << 16 ) + ( unsigned( a ) << 24 ); }

    static constexpr Color white() noexcept { return Color( 255,255,255,255 ); }
    static constexpr Color black() noexcept { return Color( 0, 0, 0, 255 ); }
    static constexpr Color gray() noexcept { return Color( 127, 127, 127, 255 ); }
    static constexpr Color red() noexcept { return Color( 255, 0, 0, 255 ); }
    static constexpr Color green() noexcept { return Color( 0, 255, 0, 255 ); }
    static constexpr Color blue() noexcept { return Color( 0, 0, 255, 255 ); }
    static constexpr Color yellow() noexcept { return Color( 255, 255, 0, 255 ); }
    static constexpr Color brown() noexcept { return Color( 135, 74, 43, 255 ); }
    static constexpr Color purple() noexcept { return Color( 128, 0, 128, 255 ); }
    static constexpr Color transparent() noexcept { return Color( 0, 0, 0, 0 ); }

    template<typename T>
    static constexpr uint8_t valToUint8( T val ) noexcept
    {
        if constexpr ( std::is_same_v<T, uint8_t> )
        {
            return val;
        }
        else if constexpr ( std::is_integral_v<T> )
        {
            return val >= 255 ? uint8_t( 255 ) : ( val <= 0 ? uint8_t( 0 ) : uint8_t( val ) );
        }
        else
        {
            return val >= T( 1 ) ? uint8_t( 255 ) : ( val <= T( 0 ) ? uint8_t( 0 ) : uint8_t( val * 255 ) );
        }
    }

    template<typename T>
    explicit constexpr Color( const Vector3<T>& vec ) noexcept :
        r{ valToUint8( vec.x ) },
        g{ valToUint8( vec.y ) },
        b{ valToUint8( vec.z ) },
        a{ uint8_t( 255 ) }
    {}

    template<typename T>
    explicit constexpr Color( const Vector4<T>& vec ) noexcept :
        r{ valToUint8( vec.x ) },
        g{ valToUint8( vec.y ) },
        b{ valToUint8( vec.z ) },
        a{ valToUint8( vec.w ) }
    {}

    template<typename T>
    explicit constexpr operator Vector4<T>() const noexcept
    {
        if constexpr ( std::is_integral_v<T> )
        {
            return Vector4<T>( T( r ), T( g ), T( b ), T( a ) );
        }
        else
        {
            return Vector4<T>( T( r ), T( g ), T( b ), T( a ) ) / T( 255 );
        }
    }

    const uint8_t& operator []( int e ) const { return *( &r + e ); }
    uint8_t& operator []( int e ) { return *( &r + e ); }

    Color& operator += ( const Color& other ) { *this = Color( Vector4i( *this ) + Vector4i( other ) ); return *this; }
    Color& operator -= ( const Color& other ) { *this = Color( Vector4i( *this ) - Vector4i( other ) ); return *this; }
    Color& operator *= ( float m ) { *this = Color( m * Vector4f( *this ) ); return *this; }
    Color& operator /= ( float m ) { return *this *= 1.0f / m; }

    constexpr Color scaledAlpha( float m ) const noexcept
    {
        return Color( r, g, b, uint8_t( std::clamp( m * a, 0.0f , 255.0f ) ) );
    }
};

inline bool operator ==( const Color& a, const Color& b )
{
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

inline bool operator !=( const Color& a, const Color& b )
{
    return !( a == b );
}

inline Color operator +( const Color& a, const Color& b )
{
    return Color(Vector4i( a ) + Vector4i( b ));
}

inline Color operator -( const Color& a, const Color& b )
{
    return Color( Vector4i( a ) - Vector4i( b ) );
}

inline Color operator *( float a, const Color& b )
{
    return Color( a * Vector4f( b ) );
}

inline Color operator *( const Color& b, float a )
{
    return Color( a * Vector4f( b ) );
}

inline Color operator /( const Color& b, float a )
{
    return b * ( 1 / a );
}

/// Blend two colors together
/// \note This operation is not commutative
MRMESH_API Color blend( const Color& front, const Color& back );

}
