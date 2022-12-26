#pragma once

#include <cmath>
#include "MRVector3.h"

namespace MR
{

/// three-dimensional vector
/// \ingroup VectorGroup
template <typename T>
struct Vector4
{
    using ValueType = T;
    using MatrixType = Matrix4<T>;
    static constexpr int elements = 4;

    T x = 0, y = 0, z = 0, w = 0;

    constexpr Vector4() noexcept = default;
    constexpr Vector4( T x, T y, T z, T w ) noexcept : x( x ), y( y ), z( z ), w( w )
    {
    }
    static constexpr Vector4 diagonal( T a ) noexcept
    {
        return Vector4( a, a, a, a );
    }
    template <typename U>
    constexpr explicit Vector4( const Vector4<U> & v ) noexcept : x( T( v.x ) ), y( T( v.y ) ), z( T( v.z ) ), w( T( v.w ) ) 
    { 
    }

    constexpr const T & operator []( int e ) const noexcept { return *( &x + e ); }
    constexpr       T & operator []( int e )       noexcept { return *( &x + e ); }

    T lengthSq() const
    {
        return x * x + y * y + z * z + w * w;
    }
    T length() const
    {
        return std::sqrt( lengthSq() );
    }

    Vector4 normalized() const
    {
        auto len = length();
        if ( len <= 0 )
            return {};
        return ( 1 / len ) * ( *this );
    }

    Vector4 operator -() const { return Vector4( -x, -y, -z, -w ); }
    const Vector4 & operator +() const { return *this; }

    Vector4 & operator +=( const Vector4<T> & b )
    {
        x += b.x; y += b.y; z += b.z; w += b.w; return *this;
    }
    Vector4 & operator -=( const Vector4<T> & b )
    {
        x -= b.x; y -= b.y; z -= b.z; w -= b.w; return *this;
    }
    Vector4 & operator *=( T b ) { x *= b; y *= b; z *= b; w *= b; return * this; }
    Vector4 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { x /= b; y /= b; z /= b; w /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }

    /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
    Vector3<T> proj3d() const
    {
        return { x / w, y / w, z / w };
    }
};

/// \related Vector4
/// \{

template <typename T>
inline bool operator ==( const Vector4<T> & a, const Vector4<T> & b )
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <typename T>
inline bool operator !=( const Vector4<T> & a, const Vector4<T> & b )
{
    return !( a == b );
}

template <typename T>
inline Vector4<T> operator +( const Vector4<T> & a, const Vector4<T> & b )
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <typename T>
inline Vector4<T> operator -( const Vector4<T> & a, const Vector4<T> & b )
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <typename T>
inline Vector4<T> operator *( T a, const Vector4<T> & b )
{
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

template <typename T> 
inline Vector4<T> operator *( const Vector4<T> & b, T a )
{ 
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

template <typename T> 
inline Vector4<T> operator /( Vector4<T> b, T a )
    { b /= a; return b; }


/// dot product
template <typename T>
inline T dot( const Vector4<T>& a, const Vector4<T>& b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <typename T> 
inline auto begin( const Vector4<T> & v ) { return &v[0]; }
template <typename T> 
inline auto begin( Vector4<T> & v ) { return &v[0]; }

template <typename T> 
inline auto end( const Vector4<T> & v ) { return &v[3]; }
template <typename T> 
inline auto end( Vector4<T> & v ) { return &v[3]; }

/// \}

} // namespace MR
