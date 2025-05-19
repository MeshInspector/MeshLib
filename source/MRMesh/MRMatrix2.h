#pragma once

#include "MRVector2.h"
#include "MRConstants.h"

namespace MR
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4804) // unsafe use of type 'bool' in operation
#pragma warning(disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

/// arbitrary 2x2 matrix
/// \ingroup MatrixGroup
template <typename T>
struct Matrix2
{
    using ValueType = T;
    using VectorType = Vector2<T>;

    /// rows, identity matrix by default
    Vector2<T> x{ 1, 0 };
    Vector2<T> y{ 0, 1 };

    constexpr Matrix2() noexcept = default;
    /// initializes matrix from its 2 rows
    constexpr Matrix2( const Vector2<T> & x, const Vector2<T> & y ) : x( x ), y( y ) { }
    template <typename U>
    constexpr explicit Matrix2( const Matrix2<U> & m ) : x( m.x ), y( m.y ) { }
    static constexpr Matrix2 zero() noexcept { return Matrix2( Vector2<T>(), Vector2<T>() ); }
    static constexpr Matrix2 identity() noexcept { return Matrix2(); }
    /// returns a matrix that scales uniformly
    static constexpr Matrix2 scale( T s ) noexcept { return Matrix2( { s, T(0) }, { T(0), s } ); }
    /// returns a matrix that has its own scale along each axis
    static constexpr Matrix2 scale( T sx, T sy ) noexcept { return Matrix2( { sx, T(0) }, { T(0), sy } ); }
    static constexpr Matrix2 scale( const Vector2<T> & s ) noexcept { return Matrix2( { s.x, T(0) }, { T(0), s.y } ); }
    /// creates matrix representing rotation around origin on given angle
    static constexpr Matrix2 rotation( T angle ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// creates matrix representing rotation that after application to (from) makes (to) vector
    static constexpr Matrix2 rotation( const Vector2<T> & from, const Vector2<T> & to ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// constructs a matrix from its 2 rows
    static constexpr Matrix2 fromRows( const Vector2<T> & x, const Vector2<T> & y ) noexcept { return Matrix2( x, y ); }
    /// constructs a matrix from its 2 columns;
    /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
    static constexpr Matrix2 fromColumns( const Vector2<T> & x, const Vector2<T> & y ) noexcept { return Matrix2( x, y ).transposed(); }

    /// row access
    constexpr const Vector2<T> & operator []( int row ) const noexcept { return *( &x + row ); }
    constexpr       Vector2<T> & operator []( int row )       noexcept { return *( &x + row ); }

    /// column access
    constexpr Vector2<T> col( int i ) const noexcept { return { x[i], y[i] }; }

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return x.x + y.y; }
    /// compute sum of squared matrix elements
    constexpr T normSq() const noexcept { return x.lengthSq() + y.lengthSq(); }
    constexpr auto norm() const noexcept
    {
        // Calling `sqrt` this way to hopefully support boost.multiprecision numbers.
        // Returning `auto` to not break on integral types.
        using std::sqrt;
        return sqrt( normSq() );
    }
    /// computes determinant of the matrix
    constexpr T det() const noexcept;
    /// computes inverse matrix
    constexpr Matrix2<T> inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// computes transposed matrix
    constexpr Matrix2<T> transposed() const noexcept;

    [[nodiscard]] friend constexpr bool operator ==( const Matrix2<T> & a, const Matrix2<T> & b ) { return a.x == b.x && a.y == b.y; }
    [[nodiscard]] friend constexpr bool operator !=( const Matrix2<T> & a, const Matrix2<T> & b ) { return !( a == b ); }

    // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.

    [[nodiscard]] friend constexpr auto operator +( const Matrix2<T> & a, const Matrix2<T> & b ) -> Matrix2<decltype( std::declval<T>() + std::declval<T>() )> { return { a.x + b.x, a.y + b.y }; }
    [[nodiscard]] friend constexpr auto operator -( const Matrix2<T> & a, const Matrix2<T> & b ) -> Matrix2<decltype( std::declval<T>() - std::declval<T>() )> { return { a.x - b.x, a.y - b.y }; }
    [[nodiscard]] friend constexpr auto operator *(               T    a, const Matrix2<T> & b ) -> Matrix2<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y }; }
    [[nodiscard]] friend constexpr auto operator *( const Matrix2<T> & b,               T    a ) -> Matrix2<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y }; }
    [[nodiscard]] friend constexpr auto operator /(       Matrix2<T>   b,               T    a ) -> Matrix2<decltype( std::declval<T>() / std::declval<T>() )>
    {
        if constexpr ( std::is_integral_v<T> )
            return { b.x / a, b.y / a };
        else
            return b * ( 1 / a );
    }

    friend constexpr Matrix2<T> & operator +=( Matrix2<T> & a, const Matrix2<T> & b ) { a.x += b.x; a.y += b.y; return a; }
    friend constexpr Matrix2<T> & operator -=( Matrix2<T> & a, const Matrix2<T> & b ) { a.x -= b.x; a.y -= b.y; return a; }
    friend constexpr Matrix2<T> & operator *=( Matrix2<T> & a,               T    b ) { a.x *= b; a.y *= b; return a; }
    friend constexpr Matrix2<T> & operator /=( Matrix2<T> & a,               T    b )
    {
        if constexpr ( std::is_integral_v<T> )
            { a.x /= b; a.y /= b; return a; }
        else
            return a *= ( 1 / b );
    }

    /// x = a * b
    [[nodiscard]] friend constexpr auto operator *( const Matrix2<T> & a, const Vector2<T> & b ) -> Vector2<decltype( dot( std::declval<Vector2<T>>(), std::declval<Vector2<T>>() ) )>
    {
        return { dot( a.x, b ), dot( a.y, b ) };
    }

    /// product of two matrices
    [[nodiscard]] friend constexpr auto operator *( const Matrix2<T> & a, const Matrix2<T> & b ) -> Matrix2<decltype( dot( std::declval<Vector2<T>>(), std::declval<Vector2<T>>() ) )>
    {
        Matrix2<decltype( dot( std::declval<Vector2<T>>(), std::declval<Vector2<T>>() ) )> res;
        for ( int i = 0; i < 2; ++i )
            for ( int j = 0; j < 2; ++j )
                res[i][j] = dot( a[i], b.col(j) );
        return res;
    }
};

/// \related Matrix2
/// \{

/// double-dot product: x = a : b
template <typename T>
inline auto dot( const Matrix2<T> & a, const Matrix2<T> & b ) -> decltype( dot( a.x, b.x ) )
{
    return dot( a.x, b.x ) + dot( a.y, b.y );
}

/// x = a * b^T
template <typename T>
inline Matrix2<T> outer( const Vector2<T> & a, const Vector2<T> & b )
{
    return { a.x * b, a.y * b };
}

template <typename T>
constexpr Matrix2<T> Matrix2<T>::rotation( T angle ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    T c = cos( angle );
    T s = sin( angle );
    return {
        { c, -s },
        { s,  c }
    };
}

template <typename T>
constexpr Matrix2<T> Matrix2<T>::rotation( const Vector2<T> & from, const Vector2<T> & to ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    const auto x = cross( from, to );
    if ( x > 0 )
        return rotation( angle( from, to ) );
    if ( x < 0 )
        return rotation( -angle( from, to ) );
    if ( dot( from, to ) >= 0 )
        return {}; // identity matrix
    return rotation( T( PI ) );
}

template <typename T>
constexpr T Matrix2<T>::det() const noexcept
{
    return x.x * y.y - x.y * y.x;
}

template <typename T>
constexpr Matrix2<T> Matrix2<T>::inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    auto det = this->det();
    if ( det == 0 )
        return {};
    return Matrix2<T>
    {
        {   y.y, - x.y },
        { - y.x,   x.x }
    } / det;
}

template <typename T>
constexpr Matrix2<T> Matrix2<T>::transposed() const noexcept
{
    return Matrix2<T>
    {
        { x.x, y.x },
        { x.y, y.y }
    };
}

/// \}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace MR
