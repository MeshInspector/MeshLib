#pragma once

#include "MRMeshFwd.h"
#include <cmath>
#include <algorithm>

namespace MR
{
 
/// three-dimensional vector
/// \ingroup VectorGroup
template <typename T> 
struct Vector3
{
    using ValueType = T;
    using MatrixType = Matrix3<T>;
    using SymMatrixType = SymMatrix3<T>;
    static constexpr int elements = 3;

    T x, y, z;

    constexpr Vector3() noexcept : x( 0 ), y( 0 ), z( 0 ) { }
    constexpr Vector3( NoInit ) noexcept { }
    constexpr Vector3( T x, T y, T z ) noexcept : x( x ), y( y ), z( z ) { }
    explicit constexpr Vector3( const Vector2<T> & v ) noexcept : x( v.x ), y( v.y ) { }

    static constexpr Vector3 diagonal( T a ) noexcept { return Vector3( a, a, a ); }
    static constexpr Vector3 plusX() noexcept { return Vector3( 1, 0, 0 ); }
    static constexpr Vector3 plusY() noexcept { return Vector3( 0, 1, 0 ); }
    static constexpr Vector3 plusZ() noexcept { return Vector3( 0, 0, 1 ); }
    static constexpr Vector3 minusX() noexcept { return Vector3( -1, 0, 0 ); }
    static constexpr Vector3 minusY() noexcept { return Vector3( 0, -1, 0 ); }
    static constexpr Vector3 minusZ() noexcept { return Vector3( 0, 0, -1 ); }

    template <typename U>
    constexpr explicit Vector3( const Vector3<U> & v ) noexcept : x( T( v.x ) ), y( T( v.y ) ), z( T( v.z ) ) { }

    constexpr const T & operator []( int e ) const noexcept { return *( &x + e ); }
    constexpr       T & operator []( int e )       noexcept { return *( &x + e ); }

    T lengthSq() const { return x * x + y * y + z * z; }
    T length() const { return std::sqrt( lengthSq() ); }

    Vector3 normalized() const 
    {
        if constexpr ( std::is_floating_point_v<T> )
        {
            auto len = length();
            if ( len <= 0 )
                return {};
            return ( 1 / len ) * (*this);
        }
        else
        {
            static_assert( dependent_false<T>, "normalized makes sense for floating point vectors only" );
        }
    }

    /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
    Vector3 furthestBasisVector() const;

    /// returns 2 unit vector, which together with this vector make an orthogonal basis
    std::pair<Vector3, Vector3> perpendicular() const;

    /// returns this vector transformed by xf if it is
    Vector3 transformed( const AffineXf3<T>* xf ) const
    {
        return xf ? ( *xf )( *this ) : *this;
    }
};

/// \related Vector3
/// \{

template <typename T> 
inline Vector3<T> & operator +=( Vector3<T> & a, const Vector3<T> & b ) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }

template <typename T> 
inline Vector3<T> & operator -=( Vector3<T> & a, const Vector3<T> & b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }

template <typename T> 
inline Vector3<T> & operator *=( Vector3<T> & a, T b ) { a.x *= b; a.y *= b; a.z *= b; return a; }

template <typename T> 
inline Vector3<T> & operator /=( Vector3<T> & a, T b ) 
{
    if constexpr ( std::is_integral_v<T> )
        { a.x /= b; a.y /= b; a.z /= b; return a; }
    else
        return a *= ( 1 / b );
}

template <typename T> 
inline Vector3<T> operator -( const Vector3<T> & a ) { return Vector3<T>( -a.x, -a.y, -a.z ); }

template <typename T> 
inline const Vector3<T> & operator +( const Vector3<T> & a ) { return a; }

/// cross product
template <typename T> 
inline Vector3<T> cross( const Vector3<T> & a, const Vector3<T> & b )
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

/// dot product
template <typename T> 
inline T dot( const Vector3<T> & a, const Vector3<T> & b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// mixed product
template <typename T> 
inline T mixed( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c )
{
    return dot( a, cross( b, c ) );
}

/// per component multiplication
template <typename T>
inline Vector3<T> mult( const Vector3<T>& a, const Vector3<T>& b )
{
    return { a.x * b.x,a.y * b.y,a.z * b.z };
}


/// angle in radians between two vectors
template <typename T> 
inline T angle( const Vector3<T> & a, const Vector3<T> & b )
{
    return std::atan2( cross( a, b ).length(), dot( a, b ) );
    // this version is slower and less precise
    //return std::acos( std::clamp( dot( a.normalized(), b.normalized() ), T(-1), T(1) ) );
}

template <typename T> 
inline Vector3<T> Vector3<T>::furthestBasisVector() const
{
    if ( fabs( x ) < fabs( y ) )
        return ( fabs( x ) < fabs( z ) ) ? Vector3( 1, 0, 0 ) : Vector3( 0, 0, 1 );
    else
        return ( fabs( y ) < fabs( z ) ) ? Vector3( 0, 1, 0 ) : Vector3( 0, 0, 1 );
}

template <typename T> 
inline std::pair<Vector3<T>, Vector3<T>> Vector3<T>::perpendicular() const
{
    std::pair<Vector3<T>, Vector3<T>> res;
    auto c1 = furthestBasisVector();
    res.first  = cross( *this, c1 ).normalized();
    res.second = cross( *this, res.first ).normalized();
    return res;
}

template <typename T> 
inline bool operator ==( const Vector3<T> & a, const Vector3<T> & b )
    { return a.x == b.x && a.y == b.y && a.z == b.z; }

template <typename T> 
inline bool operator !=( const Vector3<T> & a, const Vector3<T> & b )
    { return !( a == b ); }

template <typename T> 
inline Vector3<T> operator +( const Vector3<T> & a, const Vector3<T> & b )
    { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

template <typename T> 
inline Vector3<T> operator -( const Vector3<T> & a, const Vector3<T> & b )
    { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

template <typename T> 
inline Vector3<T> operator *( T a, const Vector3<T> & b )
    { return { a * b.x, a * b.y, a * b.z }; }

template <typename T> 
inline Vector3<T> operator *( const Vector3<T> & b, T a )
    { return { a * b.x, a * b.y, a * b.z }; }

template <typename T> 
inline Vector3<T> operator /( Vector3<T> b, T a )
    { b /= a; return b; }

template <typename T> 
inline auto begin( const Vector3<T> & v ) { return &v[0]; }
template <typename T> 
inline auto begin( Vector3<T> & v ) { return &v[0]; }

template <typename T> 
inline auto end( const Vector3<T> & v ) { return &v[3]; }
template <typename T> 
inline auto end( Vector3<T> & v ) { return &v[3]; }

/// \}

} // namespace MR
