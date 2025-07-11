#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include "MRConstants.h"
#include "MRPch/MRBindingMacros.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#if MR_HAS_REQUIRES
#include <concepts>
#endif

namespace MR
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4804) // unsafe use of type 'bool' in operation
#pragma warning(disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

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

    template <typename U> MR_REQUIRES_IF_SUPPORTED( std::constructible_from<T, U> )
    explicit constexpr Vector3( const Vector2<U> & v ) noexcept : x( v.x ), y( v.y ), z( 0 ) { }

    static constexpr Vector3 diagonal( T a ) noexcept { return Vector3( a, a, a ); }
    static constexpr Vector3 plusX() noexcept { return Vector3( 1, 0, 0 ); }
    static constexpr Vector3 plusY() noexcept { return Vector3( 0, 1, 0 ); }
    static constexpr Vector3 plusZ() noexcept { return Vector3( 0, 0, 1 ); }
    static constexpr Vector3 minusX() noexcept { return Vector3( -1, 0, 0 ); }
    static constexpr Vector3 minusY() noexcept { return Vector3( 0, -1, 0 ); }
    static constexpr Vector3 minusZ() noexcept { return Vector3( 0, 0, -1 ); }

    // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
    //   when generating the bindings, and looks out of place there.
    template <typename U> MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, U> )
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
    template <MR_SAME_TYPE_TEMPLATE_PARAM(T, TT)> // Need this, otherwise the bindings try to instantiate `AffineXf3` with non-FP arguments.
    Vector3 transformed( const AffineXf3<TT>* xf ) const MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
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

    [[nodiscard]] friend constexpr bool operator ==( const Vector3<T> & a, const Vector3<T> & b ) { return a.x == b.x && a.y == b.y && a.z == b.z; }
    [[nodiscard]] friend constexpr bool operator !=( const Vector3<T> & a, const Vector3<T> & b ) { return !( a == b ); }

    // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.

    [[nodiscard]] friend constexpr const Vector3<T> & operator +( const Vector3<T> & a ) { return a; }
    [[nodiscard]] friend constexpr auto operator -( const Vector3<T> & a ) -> Vector3<decltype( -std::declval<T>() )> { return { -a.x, -a.y, -a.z }; }

    [[nodiscard]] friend constexpr auto operator +( const Vector3<T> & a, const Vector3<T> & b ) -> Vector3<decltype( std::declval<T>() + std::declval<T>() )> { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
    [[nodiscard]] friend constexpr auto operator -( const Vector3<T> & a, const Vector3<T> & b ) -> Vector3<decltype( std::declval<T>() - std::declval<T>() )> { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
    [[nodiscard]] friend constexpr auto operator *(               T    a, const Vector3<T> & b ) -> Vector3<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z }; }
    [[nodiscard]] friend constexpr auto operator *( const Vector3<T> & b,               T    a ) -> Vector3<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z }; }
    [[nodiscard]] friend constexpr auto operator /(       Vector3<T>   b,               T    a ) -> Vector3<decltype( std::declval<T>() / std::declval<T>() )>
    {
        if constexpr ( std::is_integral_v<T> )
            return { b.x / a, b.y / a, b.z / a };
        else
            return b * ( 1 / a );
    }

    friend constexpr Vector3<T> & operator +=( Vector3<T> & a, const Vector3<T> & b ) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    friend constexpr Vector3<T> & operator -=( Vector3<T> & a, const Vector3<T> & b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
    friend constexpr Vector3<T> & operator *=( Vector3<T> & a,               T    b ) { a.x *= b; a.y *= b; a.z *= b; return a; }
    friend constexpr Vector3<T> & operator /=( Vector3<T> & a,               T    b )
    {
        if constexpr ( std::is_integral_v<T> )
            { a.x /= b; a.y /= b; a.z /= b; return a; }
        else
            return a *= ( 1 / b );
    }
};

/// \related Vector3
/// \{


/// squared distance between two points, which is faster to compute than just distance
template <typename T>
inline T distanceSq( const Vector3<T> & a, const Vector3<T> & b )
{
    return ( a - b ).lengthSq();
}

/// distance between two points, better use distanceSq for higher performance
template <typename T>
inline T distance( const Vector3<T> & a, const Vector3<T> & b )
{
    return ( a - b ).length();
}

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
inline auto dot( const Vector3<T> & a, const Vector3<T> & b ) -> decltype( a.x * b.x )
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


// We don't need to bind those functions themselves. This doesn't prevent `__iter__` from being generated for the type.

template <typename T>
MR_BIND_IGNORE inline auto begin( const Vector3<T> & v ) { return &v[0]; }
template <typename T>
MR_BIND_IGNORE inline auto begin( Vector3<T> & v ) { return &v[0]; }

template <typename T>
MR_BIND_IGNORE inline auto end( const Vector3<T> & v ) { return &v[3]; }
template <typename T>
MR_BIND_IGNORE inline auto end( Vector3<T> & v ) { return &v[3]; }

/// \}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace MR

template<>
struct std::hash<MR::Vector3f>
{
    size_t operator()( MR::Vector3f const& p ) const noexcept
    {
        // standard implementation is slower:
        // phmap::HashState().combine(phmap::Hash<float>()(p.x), p.y, p.z);
        std::uint64_t xy;
        std::uint32_t z;
        static_assert( sizeof( float ) == sizeof( std::uint32_t ) );
        std::memcpy( &xy, &p.x, sizeof( std::uint64_t ) );
        std::memcpy( &z, &p.z, sizeof( std::uint32_t ) );
        return size_t( xy ) ^ ( size_t( z ) << 16 );
    }
};
