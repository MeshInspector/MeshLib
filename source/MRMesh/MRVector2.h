#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <utility>

namespace MR
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4804) // unsafe use of type 'bool' in operation
#pragma warning(disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

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
    explicit Vector2( NoInit ) noexcept { }
    constexpr Vector2( T x, T y ) noexcept : x( x ), y( y ) { }

    template <typename U> MR_REQUIRES_IF_SUPPORTED( std::constructible_from<T, U> )
    explicit constexpr Vector2( const Vector3<U> & v ) noexcept : x( v.x ), y( v.y ) { }

    static constexpr Vector2 diagonal( T a ) noexcept { return Vector2( a, a ); }
    static constexpr Vector2 plusX() noexcept { return Vector2( 1, 0 ); }
    static constexpr Vector2 plusY() noexcept { return Vector2( 0, 1 ); }
    static constexpr Vector2 minusX() noexcept { return Vector2( -1, 0 ); }
    static constexpr Vector2 minusY() noexcept { return Vector2( 0, -1 ); }

    // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
    //   when generating the bindings, and looks out of place there.
    template <typename U> MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, U> )
    constexpr explicit Vector2( const Vector2<U> & v ) noexcept : x( T( v.x ) ), y( T( v.y ) ) { }

    constexpr const T & operator []( int e ) const noexcept { return *( &x + e ); }
    constexpr       T & operator []( int e )       noexcept { return *( &x + e ); }

    T lengthSq() const { return x * x + y * y; }
    auto length() const
    {
        // Calling `sqrt` this way to hopefully support boost.multiprecision numbers.
        // Returning `auto` to not break on integral types.
        using std::sqrt;
        return sqrt( lengthSq() );
    }

    Vector2 normalized() const MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
    {
        auto len = length();
        if ( len <= 0 )
            return {};
        return ( 1 / len ) * (*this);
    }

    /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
    Vector2 furthestBasisVector() const MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, bool> );

    /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
    constexpr Vector2 perpendicular() const MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, bool> ) { return Vector2{ -y, x }; }

    [[nodiscard]] bool isFinite() const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
    {
        return std::isfinite( x ) && std::isfinite( y );
    }

    [[nodiscard]] friend constexpr bool operator ==( const Vector2<T> & a, const Vector2<T> & b ) { return a.x == b.x && a.y == b.y; }
    [[nodiscard]] friend constexpr bool operator !=( const Vector2<T> & a, const Vector2<T> & b ) { return !( a == b ); }

    // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.

    [[nodiscard]] friend constexpr const Vector2<T> & operator +( const Vector2<T> & a ) { return a; }
    [[nodiscard]] friend constexpr auto operator -( const Vector2<T> & a ) -> Vector2<decltype( -std::declval<T>() )> { return { -a.x, -a.y }; }

    [[nodiscard]] friend constexpr auto operator +( const Vector2<T> & a, const Vector2<T> & b ) -> Vector2<decltype( std::declval<T>() + std::declval<T>() )> { return { a.x + b.x, a.y + b.y }; }
    [[nodiscard]] friend constexpr auto operator -( const Vector2<T> & a, const Vector2<T> & b ) -> Vector2<decltype( std::declval<T>() - std::declval<T>() )> { return { a.x - b.x, a.y - b.y }; }
    [[nodiscard]] friend constexpr auto operator *(               T    a, const Vector2<T> & b ) -> Vector2<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y }; }
    [[nodiscard]] friend constexpr auto operator *( const Vector2<T> & b,               T    a ) -> Vector2<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y }; }
    [[nodiscard]] friend constexpr auto operator /(       Vector2<T>   b,               T    a ) -> Vector2<decltype( std::declval<T>() / std::declval<T>() )>
    {
        if constexpr ( std::is_integral_v<T> )
            return { b.x / a, b.y / a };
        else
            return b * ( 1 / a );
    }

    friend constexpr Vector2<T> & operator +=( Vector2<T> & a, const Vector2<T> & b ) { a.x += b.x; a.y += b.y; return a; }
    friend constexpr Vector2<T> & operator -=( Vector2<T> & a, const Vector2<T> & b ) { a.x -= b.x; a.y -= b.y; return a; }
    friend constexpr Vector2<T> & operator *=( Vector2<T> & a,               T    b ) { a.x *= b; a.y *= b; return a; }
    friend constexpr Vector2<T> & operator /=( Vector2<T> & a,               T    b )
    {
        if constexpr ( std::is_integral_v<T> )
            { a.x /= b; a.y /= b; return a; }
        else
            return a *= ( 1 / b );
    }
};

/// \related Vector2
/// \{

/// squared distance between two points, which is faster to compute than just distance
template <typename T>
inline T distanceSq( const Vector2<T> & a, const Vector2<T> & b )
{
    return ( a - b ).lengthSq();
}

/// distance between two points, better use distanceSq for higher performance
template <typename T>
inline T distance( const Vector2<T> & a, const Vector2<T> & b )
{
    return ( a - b ).length();
}

/// cross product
template <typename T>
inline T cross( const Vector2<T> & a, const Vector2<T> & b )
{
    return a.x * b.y - a.y * b.x;
}

/// dot product
template <typename T>
inline auto dot( const Vector2<T> & a, const Vector2<T> & b ) -> decltype( a.x * b.x )
{
    return a.x * b.x + a.y * b.y;
}

/// squared length
template <typename T>
inline T sqr( const Vector2<T> & a )
{
    return a.lengthSq();
}

/// per component multiplication
template <typename T>
inline Vector2<T> mult( const Vector2<T>& a, const Vector2<T>& b )
{
    return { a.x * b.x,a.y * b.y };
}

/// per component division
template <typename T>
inline Vector2<T> div( const Vector2<T>& a, const Vector2<T>& b )
{
    return { a.x / b.x, a.y / b.y };
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
inline Vector2<T> Vector2<T>::furthestBasisVector() const MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, bool> )
{
    using std::abs; // This allows boost.multiprecision numbers.
    if ( abs( x ) < abs( y ) )
        return Vector2( 1, 0 );
    else
        return Vector2( 0, 1 );
}


// We don't need to bind those functions themselves. This doesn't prevent `__iter__` from being generated for the type.

template <typename T>
MR_BIND_IGNORE inline auto begin( const Vector2<T> & v ) { return &v[0]; }
template <typename T>
MR_BIND_IGNORE inline auto begin( Vector2<T> & v ) { return &v[0]; }

template <typename T>
MR_BIND_IGNORE inline auto end( const Vector2<T> & v ) { return &v[2]; }
template <typename T>
MR_BIND_IGNORE inline auto end( Vector2<T> & v ) { return &v[2]; }

/// \}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace MR

template<>
struct std::hash<MR::Vector2f>
{
    size_t operator()( MR::Vector2f const& p ) const noexcept
    {
        std::uint64_t xy;
        static_assert( sizeof( float ) == sizeof( std::uint32_t ) );
        std::memcpy( &xy, &p.x, sizeof( std::uint64_t ) );
        return size_t( xy );
    }
};
