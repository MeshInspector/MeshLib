#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include "MRConstants.h"
#include <algorithm>
#include <cmath>
#if MR_HAS_REQUIRES
#include <concepts>
#endif

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
    explicit Vector3( NoInit ) noexcept { }
    constexpr Vector3( T x, T y, T z ) noexcept : x( x ), y( y ), z( z ) { }
    explicit constexpr Vector3( const Vector2<T> & v ) noexcept : x( v.x ), y( v.y ), z( 0 ) { }

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
    auto length() const
    {
        // Calling `sqrt` this way to hopefully support boost.multiprecision numbers.
        // Returning `auto` to not break on integral types.
        using std::sqrt;
        return sqrt( lengthSq() );
    }

    [[nodiscard]] Vector3 normalized() const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
    {
        auto len = length();
        if ( len <= 0 )
            return {};
        return ( 1 / len ) * (*this);
    }

    /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
    Vector3 furthestBasisVector() const MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, bool> );

    /// returns 2 unit vector, which together with this vector make an orthogonal basis
    /// Currently not implemented for integral vectors.
    std::pair<Vector3, Vector3> perpendicular() const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> );

    /// returns this vector transformed by xf if it is
    Vector3 transformed( const AffineXf3<T>* xf ) const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
    {
        return xf ? ( *xf )( *this ) : *this;
    }

    /// get rid of signed zero values to be sure that equal vectors have identical binary representation
    void unsignZeroValues() MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
    {
        for ( auto i = 0; i < elements; ++i )
            if ( (*this)[i] == 0.f && std::signbit( (*this)[i] ) )
                (*this)[i] = 0.f;
    }

    [[nodiscard]] bool isFinite() const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
    {
        return std::isfinite( x ) && std::isfinite( y ) && std::isfinite( z );
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

/// squared length
template <typename T>
inline T sqr( const Vector3<T> & a )
{
    return a.lengthSq();
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

/// per component division
template <typename T>
inline Vector3<T> div( const Vector3<T>& a, const Vector3<T>& b )
{
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}


/// computes minimal angle in [0,pi] between two vectors;
/// the function is symmetric: angle( a, b ) == angle( b, a )
template <typename T>
inline T angle( const Vector3<T> & a, const Vector3<T> & b )
{
    return std::atan2( cross( a, b ).length(), dot( a, b ) );
    // this version is slower and less precise
    //return std::acos( std::clamp( dot( a.normalized(), b.normalized() ), T(-1), T(1) ) );
}

template <typename T>
inline Vector3<T> Vector3<T>::furthestBasisVector() const MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, bool> )
{
    using std::abs; // This should allow boost.multiprecision numbers here.
    if ( abs( x ) < abs( y ) )
        return ( abs( x ) < abs( z ) ) ? Vector3( 1, 0, 0 ) : Vector3( 0, 0, 1 );
    else
        return ( abs( y ) < abs( z ) ) ? Vector3( 0, 1, 0 ) : Vector3( 0, 0, 1 );
}

template <typename T>
inline std::pair<Vector3<T>, Vector3<T>> Vector3<T>::perpendicular() const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
{
    std::pair<Vector3<T>, Vector3<T>> res;
    auto c1 = furthestBasisVector();
    res.first  = cross( *this, c1 ).normalized();
    res.second = cross( *this, res.first ).normalized();
    return res;
}

template <typename T>
[[nodiscard]] inline bool operator ==( const Vector3<T> & a, const Vector3<T> & b )
    { return a.x == b.x && a.y == b.y && a.z == b.z; }

template <typename T>
[[nodiscard]] inline bool operator !=( const Vector3<T> & a, const Vector3<T> & b )
    { return !( a == b ); }

template <typename T>
[[nodiscard]] inline constexpr Vector3<T> operator +( const Vector3<T> & a, const Vector3<T> & b )
    { return { T( a.x + b.x ), T( a.y + b.y ), T( a.z + b.z ) }; }

template <typename T>
[[nodiscard]] inline Vector3<T> operator -( const Vector3<T> & a, const Vector3<T> & b )
    { return { T( a.x - b.x ), T( a.y - b.y ), T( a.z - b.z ) }; }

template <typename T>
[[nodiscard]] inline Vector3<T> operator *( T a, const Vector3<T> & b )
    { return { T( a * b.x ), T( a * b.y ), T( a * b.z ) }; }

template <typename T>
[[nodiscard]] inline Vector3<T> operator *( const Vector3<T> & b, T a )
    { return { T( a * b.x ), T( a * b.y ), T( a * b.z ) }; }

template <typename T>
[[nodiscard]] inline Vector3<T> operator /( Vector3<T> b, T a )
    { b /= a; return b; }

/// returns a point on unit sphere given two angles
template <typename T>
Vector3<T> unitVector3( T azimuth, T altitude )
{
    const auto zenithAngle = T( PI2 ) - altitude;
    return
    {
        std::sin( zenithAngle ) * std::cos( azimuth ),
        std::sin( zenithAngle ) * std::sin( azimuth ),
        std::cos( zenithAngle )
    };
}

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
