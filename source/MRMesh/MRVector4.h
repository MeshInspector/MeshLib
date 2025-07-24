#pragma once

#include <cmath>
#include <type_traits>
#include "MRPch/MRBindingMacros.h"
#include "MRMesh/MRMacros.h"
#include "MRVector3.h"

namespace MR
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4804) // unsafe use of type 'bool' in operation
#pragma warning(disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

/// four-dimensional vector
/// \ingroup VectorGroup
template <typename T>
struct Vector4
{
    using ValueType = T;
    using MatrixType = Matrix4<T>;
    using SymMatrixType = SymMatrix4<T>;
    static constexpr int elements = 4;

    T x, y, z, w;

    constexpr Vector4() noexcept : x( 0 ), y( 0 ), z( 0 ), w( 0 ) { }
    explicit Vector4( NoInit ) noexcept { }
    constexpr Vector4( T x, T y, T z, T w ) noexcept : x( x ), y( y ), z( z ), w( w ) { }
    static constexpr Vector4 diagonal( T a ) noexcept
    {
        return Vector4( a, a, a, a );
    }

    // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
    //   when generating the bindings, and looks out of place there. Specifically for Vector4, it only gets emitted on Windows (but not on Linux) for some reason.
    template <typename U> MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, U> )
    constexpr explicit Vector4( const Vector4<U> & v ) noexcept : x( T( v.x ) ), y( T( v.y ) ), z( T( v.z ) ), w( T( v.w ) )
    {
    }

    constexpr const T & operator []( int e ) const noexcept { return *( &x + e ); }
    constexpr       T & operator []( int e )       noexcept { return *( &x + e ); }

    T lengthSq() const
    {
        return x * x + y * y + z * z + w * w;
    }
    auto length() const
    {
        // Calling `sqrt` this way to hopefully support boost.multiprecision numbers.
        // Returning `auto` to not break on integral types.
        using std::sqrt;
        return sqrt( lengthSq() );
    }

    Vector4 normalized() const MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
    {
        auto len = length();
        if ( len <= 0 )
            return {};
        return ( 1 / len ) * ( *this );
    }

    /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
    Vector3<T> proj3d() const MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
    {
        return { x / w, y / w, z / w };
    }

    [[nodiscard]] bool isFinite() const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
    {
        return std::isfinite( x ) && std::isfinite( y ) && std::isfinite( z ) && std::isfinite( w );
    }

    [[nodiscard]] friend constexpr bool operator ==( const Vector4<T> & a, const Vector4<T> & b ) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
    [[nodiscard]] friend constexpr bool operator !=( const Vector4<T> & a, const Vector4<T> & b ) { return !( a == b ); }

    // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.

    [[nodiscard]] friend constexpr const Vector4<T> & operator +( const Vector4<T> & a ) { return a; }
    [[nodiscard]] friend constexpr auto operator -( const Vector4<T> & a ) -> Vector4<decltype( -std::declval<T>() )> { return { -a.x, -a.y, -a.z, -a.w }; }

    [[nodiscard]] friend constexpr auto operator +( const Vector4<T> & a, const Vector4<T> & b ) -> Vector4<decltype( std::declval<T>() + std::declval<T>() )> { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
    [[nodiscard]] friend constexpr auto operator -( const Vector4<T> & a, const Vector4<T> & b ) -> Vector4<decltype( std::declval<T>() - std::declval<T>() )> { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
    [[nodiscard]] friend constexpr auto operator *(               T    a, const Vector4<T> & b ) -> Vector4<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z, a * b.w }; }
    [[nodiscard]] friend constexpr auto operator *( const Vector4<T> & b,               T    a ) -> Vector4<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z, a * b.w }; }
    [[nodiscard]] friend constexpr auto operator /(       Vector4<T>   b,               T    a ) -> Vector4<decltype( std::declval<T>() / std::declval<T>() )>
    {
        if constexpr ( std::is_integral_v<T> )
            return { b.x / a, b.y / a, b.z / a, b.w / a };
        else
            return b * ( 1 / a );
    }

    friend constexpr Vector4<T> & operator +=( Vector4<T> & a, const Vector4<T> & b ) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
    friend constexpr Vector4<T> & operator -=( Vector4<T> & a, const Vector4<T> & b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
    friend constexpr Vector4<T> & operator *=( Vector4<T> & a,               T    b ) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
    friend constexpr Vector4<T> & operator /=( Vector4<T> & a,               T    b )
    {
        if constexpr ( std::is_integral_v<T> )
            { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }
        else
            return a *= ( 1 / b );
    }
};

/// \related Vector4
/// \{

/// squared distance between two points, which is faster to compute than just distance
template <typename T>
inline T distanceSq( const Vector4<T> & a, const Vector4<T> & b )
{
    return ( a - b ).lengthSq();
}

/// distance between two points, better use distanceSq for higher performance
template <typename T>
inline T distance( const Vector4<T> & a, const Vector4<T> & b )
{
    return ( a - b ).length();
}

/// dot product
template <typename T>
inline auto dot( const Vector4<T> & a, const Vector4<T> & b ) -> decltype( a.x * b.x )
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/// squared length
template <typename T>
inline T sqr( const Vector4<T> & a )
{
    return a.lengthSq();
}

/// per component multiplication
template <typename T>
inline Vector4<T> mult( const Vector4<T>& a, const Vector4<T>& b )
{
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

/// per component division
template <typename T>
inline Vector4<T> div( const Vector4<T>& a, const Vector4<T>& b )
{
    return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}


// We don't need to bind those functions themselves. This doesn't prevent `__iter__` from being generated for the type.

template <typename T>
MR_BIND_IGNORE auto begin( const Vector4<T> & v ) { return &v[0]; }
template <typename T>
MR_BIND_IGNORE auto begin( Vector4<T> & v ) { return &v[0]; }

template <typename T>
MR_BIND_IGNORE auto end( const Vector4<T> & v ) { return &v[4]; }
template <typename T>
MR_BIND_IGNORE auto end( Vector4<T> & v ) { return &v[4]; }

/// \}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace MR
