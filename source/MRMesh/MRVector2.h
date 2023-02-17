#pragma once

#include "MRMeshFwd.h"
#include <cmath>
#include <algorithm>

namespace MR
{

/// \defgroup VectorGroup Vector
/// \ingroup MathGroup
 
/// two-dimensional vector
/// \ingroup VectorGroup
template <typename T> 
struct Vector2
{
    using ValueType = T;
    using MatrixType = Matrix2<T>;
    using SymMatrixType = SymMatrix2<T>;
    static constexpr int elements = 2;

    T x, y;

    constexpr Vector2() noexcept : x( 0 ), y( 0 ) { }
    explicit constexpr Vector2( NoInit ) noexcept { }
    constexpr Vector2( T x, T y ) noexcept : x( x ), y( y ) { }
    explicit constexpr Vector2( const Vector3<T> & v ) noexcept : x( v.x ), y( v.y ) { }

    static constexpr Vector2 diagonal( T a ) noexcept { return Vector2( a, a ); }
    static constexpr Vector2 plusX() noexcept { return Vector2( 1, 0 ); }
    static constexpr Vector2 plusY() noexcept { return Vector2( 0, 1 ); }
    static constexpr Vector2 minusX() noexcept { return Vector2( -1, 0 ); }
    static constexpr Vector2 minusY() noexcept { return Vector2( 0, -1 ); }

    template <typename U>
    constexpr explicit Vector2( const Vector2<U> & v ) noexcept : x( T( v.x ) ), y( T( v.y ) ) { }

    constexpr const T & operator []( int e ) const noexcept { return *( &x + e ); }
    constexpr       T & operator []( int e )       noexcept { return *( &x + e ); }

    T lengthSq() const { return x * x + y * y; }
    T length() const { return T( std::sqrt( lengthSq() ) ); }

    Vector2 normalized() const 
    {
        auto len = length();
        if ( len <= 0 )
            return {};
        return ( 1 / len ) * (*this);
    }

    Vector2 operator -() const { return Vector2( -x, -y ); }
    const Vector2 & operator +() const { return *this; }

    /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
    Vector2 furthestBasisVector() const;

    /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
    Vector2 perpendicular() const { return Vector2{ -y, x }; }

    Vector2 & operator +=( const Vector2<T> & b ) { x += b.x; y += b.y; return * this; }
    Vector2 & operator -=( const Vector2<T> & b ) { x -= b.x; y -= b.y; return * this; }
    Vector2 & operator *=( T b ) { x *= b; y *= b; return * this; }
    Vector2 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { x /= b; y /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }

    T& u() { return x; }
    T& v() { return y; }
};

/// \related Vector2
/// \{

/// cross product
template <typename T> 
inline T cross( const Vector2<T> & a, const Vector2<T> & b )
{
    return a.x * b.y - a.y * b.x;
}

/// dot product
template <typename T> 
inline T dot( const Vector2<T> & a, const Vector2<T> & b )
{
    return a.x * b.x + a.y * b.y;
}

/// angle in radians between two vectors
template <typename T> 
inline T angle( const Vector2<T> & a, const Vector2<T> & b )
{
    return std::atan2( std::abs( cross( a, b ) ), dot( a, b ) );
    // this version is slower and less precise
    //return std::acos( std::clamp( dot( a.normalized(), b.normalized() ), T(-1), T(1) ) );
}

template <typename T> 
inline Vector2<T> Vector2<T>::furthestBasisVector() const
{
    if ( fabs( x ) < fabs( y ) )
        return Vector2( 1, 0 );
    else
        return Vector2( 0, 1 );
}

template <typename T> 
inline bool operator ==( const Vector2<T> & a, const Vector2<T> & b )
    { return a.x == b.x && a.y == b.y; }

template <typename T> 
inline bool operator !=( const Vector2<T> & a, const Vector2<T> & b )
    { return !( a == b ); }

template <typename T> 
inline Vector2<T> operator +( const Vector2<T> & a, const Vector2<T> & b )
    { return { a.x + b.x, a.y + b.y }; }

template <typename T> 
inline Vector2<T> operator -( const Vector2<T> & a, const Vector2<T> & b )
    { return { a.x - b.x, a.y - b.y }; }

template <typename T> 
inline Vector2<T> operator *( T a, const Vector2<T> & b )
    { return { a * b.x, a * b.y }; }

template <typename T> 
inline Vector2<T> operator *( const Vector2<T> & b, T a )
    { return { a * b.x, a * b.y }; }

template <typename T> 
inline Vector2<T> operator /( Vector2<T> b, T a )
    { b /= a; return b; }

template <typename T> 
inline auto begin( const Vector2<T> & v ) { return &v[0]; }
template <typename T> 
inline auto begin( Vector2<T> & v ) { return &v[0]; }

template <typename T> 
inline auto end( const Vector2<T> & v ) { return &v[2]; }
template <typename T> 
inline auto end( Vector2<T> & v ) { return &v[2]; }

/// \}

} // namespace MR
