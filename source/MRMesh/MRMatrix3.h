#pragma once

#include "MRVector3.h"
#include "MRConstants.h"

namespace MR
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4804) // unsafe use of type 'bool' in operation
#pragma warning(disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

/// arbitrary 3x3 matrix
/// \ingroup MatrixGroup
template <typename T>
struct Matrix3
{
    using ValueType = T;
    using VectorType = Vector3<T>;

    /// rows, identity matrix by default
    Vector3<T> x{ 1, 0, 0 };
    Vector3<T> y{ 0, 1, 0 };
    Vector3<T> z{ 0, 0, 1 };

    constexpr Matrix3() noexcept = default;
    /// initializes matrix from its 3 rows
    constexpr Matrix3( const Vector3<T> & x, const Vector3<T> & y, const Vector3<T> & z ) : x( x ), y( y ), z( z ) { }
    template <typename U>
    constexpr explicit Matrix3( const Matrix3<U> & m ) : x( m.x ), y( m.y ), z( m.z ) { }
    static constexpr Matrix3 zero() noexcept { return Matrix3( Vector3<T>(), Vector3<T>(), Vector3<T>() ); }
    static constexpr Matrix3 identity() noexcept { return Matrix3(); }
    /// returns a matrix that scales uniformly
    static constexpr Matrix3 scale( T s ) noexcept { return Matrix3( { s, T(0), T(0) }, { T(0), s, T(0) }, { T(0), T(0), s } ); }
    /// returns a matrix that has its own scale along each axis
    static constexpr Matrix3 scale( T sx, T sy, T sz ) noexcept { return Matrix3( { sx, T(0), T(0) }, { T(0), sy, T(0) }, { T(0), T(0), sz } ); }
    static constexpr Matrix3 scale( const Vector3<T> & s ) noexcept { return Matrix3( { s.x, T(0), T(0) }, { T(0), s.y, T(0) }, { T(0), T(0), s.z } ); }
    /// creates matrix representing rotation around given axis on given angle
    static constexpr Matrix3 rotation( const Vector3<T> & axis, T angle ) noexcept MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> );
    /// creates matrix representing rotation that after application to (from) makes (to) vector
    static constexpr Matrix3 rotation( const Vector3<T> & from, const Vector3<T> & to ) noexcept MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> );
    /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
    /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
    static constexpr Matrix3 rotationFromEuler( const Vector3<T> & eulerAngles ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// returns linear by angles approximation of the rotation matrix, which is close to true rotation matrix for small angles
    static constexpr Matrix3 approximateLinearRotationMatrixFromEuler( const Vector3<T> & eulerAngles ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// constructs a matrix from its 3 rows
    static constexpr Matrix3 fromRows( const Vector3<T> & x, const Vector3<T> & y, const Vector3<T> & z ) noexcept { return Matrix3( x, y, z ); }
    /// constructs a matrix from its 3 columns;
    /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
    static constexpr Matrix3 fromColumns( const Vector3<T> & x, const Vector3<T> & y, const Vector3<T> & z ) noexcept { return Matrix3( x, y, z ).transposed(); }

    /// row access
    constexpr const Vector3<T> & operator []( int row ) const noexcept { return *( &x + row ); }
    constexpr       Vector3<T> & operator []( int row )       noexcept { return *( &x + row ); }

    /// column access
    constexpr Vector3<T> col( int i ) const noexcept { return { x[i], y[i], z[i] }; }

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return x.x + y.y + z.z; }
    /// compute sum of squared matrix elements
    constexpr T normSq() const noexcept { return x.lengthSq() + y.lengthSq() + z.lengthSq(); }
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
    constexpr Matrix3<T> inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> );
    /// computes transposed matrix
    constexpr Matrix3<T> transposed() const noexcept;
    /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
    constexpr Vector3<T> toEulerAngles() const noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );

    struct QR
    {
        Matrix3 q, r;
    };
    /// decompose this matrix on the product Q*R, where Q is orthogonal and R is upper triangular
    QR qr() const noexcept MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> );

    [[nodiscard]] friend constexpr bool operator ==( const Matrix3<T> & a, const Matrix3<T> & b ) { return a.x == b.x && a.y == b.y && a.z == b.z; }
    [[nodiscard]] friend constexpr bool operator !=( const Matrix3<T> & a, const Matrix3<T> & b ) { return !( a == b ); }

    // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.

    [[nodiscard]] friend constexpr auto operator +( const Matrix3<T> & a, const Matrix3<T> & b ) -> Matrix3<decltype( std::declval<T>() + std::declval<T>() )> { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
    [[nodiscard]] friend constexpr auto operator -( const Matrix3<T> & a, const Matrix3<T> & b ) -> Matrix3<decltype( std::declval<T>() - std::declval<T>() )> { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
    [[nodiscard]] friend constexpr auto operator *(               T    a, const Matrix3<T> & b ) -> Matrix3<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z }; }
    [[nodiscard]] friend constexpr auto operator *( const Matrix3<T> & b,               T    a ) -> Matrix3<decltype( std::declval<T>() * std::declval<T>() )> { return { a * b.x, a * b.y, a * b.z }; }
    [[nodiscard]] friend constexpr auto operator /(       Matrix3<T>   b,               T    a ) -> Matrix3<decltype( std::declval<T>() / std::declval<T>() )>
    {
        if constexpr ( std::is_integral_v<T> )
            return { b.x / a, b.y / a, b.z / a };
        else
            return b * ( 1 / a );
    }

    friend constexpr Matrix3<T> & operator +=( Matrix3<T> & a, const Matrix3<T> & b ) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    friend constexpr Matrix3<T> & operator -=( Matrix3<T> & a, const Matrix3<T> & b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
    friend constexpr Matrix3<T> & operator *=( Matrix3<T> & a,               T    b ) { a.x *= b; a.y *= b; a.z *= b; return a; }
    friend constexpr Matrix3<T> & operator /=( Matrix3<T> & a,               T    b )
    {
        if constexpr ( std::is_integral_v<T> )
            { a.x /= b; a.y /= b; a.z /= b; return a; }
        else
            return a *= ( 1 / b );
    }

    /// x = a * b
    [[nodiscard]] friend constexpr auto operator *( const Matrix3<T> & a, const Vector3<T> & b ) -> Vector3<decltype( dot( std::declval<Vector3<T>>(), std::declval<Vector3<T>>() ) )>
    {
        return { dot( a.x, b ), dot( a.y, b ), dot( a.z, b ) };
    }

    /// product of two matrices
    [[nodiscard]] friend constexpr auto operator *( const Matrix3<T> & a, const Matrix3<T> & b ) -> Matrix3<decltype( dot( std::declval<Vector3<T>>(), std::declval<Vector3<T>>() ) )>
    {
        Matrix3<decltype( dot( std::declval<Vector3<T>>(), std::declval<Vector3<T>>() ) )> res;
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                res[i][j] = dot( a[i], b.col(j) );
        return res;
    }
};

/// \related Matrix3
/// \{

/// double-dot product: x = a : b
template <typename T>
inline auto dot( const Matrix3<T> & a, const Matrix3<T> & b ) -> decltype( dot( a.x, b.x ) )
{
    return dot( a.x, b.x ) + dot( a.y, b.y ) + dot( a.z, b.z );
}

/// x = a * b^T
template <typename T>
inline Matrix3<T> outer( const Vector3<T> & a, const Vector3<T> & b )
{
    return { a.x * b, a.y * b, a.z * b };
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::rotation( const Vector3<T> & axis, T angle ) noexcept MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
{
    // https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    auto u = axis.normalized();
    T c = cos( angle );
    T oc = 1 - c;
    T s = sin( angle );
    return {
        {   c + u.x * u.x * oc,     u.x * u.y * oc - u.z * s, u.x * u.z * oc + u.y * s },
        { u.y * u.x * oc + u.z * s,   c + u.y * u.y * oc,     u.y * u.z * oc - u.x * s },
        { u.z * u.x * oc - u.y * s, u.z * u.y * oc + u.x * s,   c + u.z * u.z * oc     }
    };
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::rotation( const Vector3<T> & from, const Vector3<T> & to ) noexcept MR_REQUIRES_IF_SUPPORTED( std::floating_point<T> )
{
    auto axis = cross( from, to );
    if ( axis.lengthSq() > 0 )
        return rotation( axis, angle( from, to ) );
    if ( dot( from, to ) >= 0 )
        return {}; // identity matrix
    return rotation( cross( from, from.furthestBasisVector() ), T( PI ) );
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::rotationFromEuler( const Vector3<T> & eulerAngles ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // https://www.geometrictools.com/Documentation/EulerAngles.pdf (36)
    const auto cx = std::cos( eulerAngles.x );
    const auto cy = std::cos( eulerAngles.y );
    const auto cz = std::cos( eulerAngles.z );
    const auto sx = std::sin( eulerAngles.x );
    const auto sy = std::sin( eulerAngles.y );
    const auto sz = std::sin( eulerAngles.z );
    return {
        { cy * cz,   cz * sx * sy - cx * sz,   cx * cz * sy + sx * sz },
        { cy * sz,   cx * cz + sx * sy * sz,  -cz * sx + cx * sy * sz },
        {     -sy,   cy * sx,                  cx * cy                }
    };
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::approximateLinearRotationMatrixFromEuler( const Vector3<T> & eulerAngles ) noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    const auto alpha = eulerAngles.x;
    const auto  beta = eulerAngles.y;
    const auto gamma = eulerAngles.z;
    return {
        {  T(1), -gamma,   beta },
        { gamma,   T(1), -alpha },
        { -beta,  alpha,   T(1) }
    };
}

template <typename T>
constexpr T Matrix3<T>::det() const noexcept
{
    return
        x.x * ( y.y * z.z - y.z * z.y )
     -  x.y * ( y.x * z.z - y.z * z.x )
     +  x.z * ( y.x * z.y - y.y * z.x );
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
{
    auto det = this->det();
    if ( det == 0 )
        return {};
    return Matrix3<T>
    {
        { y.y * z.z - y.z * z.y,   x.z * z.y - x.y * z.z,   x.y * y.z - x.z * y.y },
        { y.z * z.x - y.x * z.z,   x.x * z.z - x.z * z.x,   x.z * y.x - x.x * y.z },
        { y.x * z.y - y.y * z.x,   x.y * z.x - x.x * z.y,   x.x * y.y - x.y * y.x }
    } / det;
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::transposed() const noexcept
{
    return Matrix3<T>
    {
        { x.x, y.x, z.x },
        { x.y, y.y, z.y },
        { x.z, y.z, z.z }
    };
}

template <typename T>
constexpr Vector3<T> Matrix3<T>::toEulerAngles() const noexcept MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    return {
        std::atan2(  z.y, z.z ),
        std::atan2( -z.x, std::sqrt( z.y * z.y + z.z * z.z ) ),
        std::atan2(  y.x, x.x )
    };
}

template <typename T>
auto Matrix3<T>::qr() const noexcept -> QR MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
{
    // https://en.wikipedia.org/wiki/QR_decomposition#Computing_the_QR_decomposition
    const auto a0 = col( 0 );
    auto a1 = col( 1 );
    auto a2 = col( 2 );
    const auto r00 = a0.length();
    const auto e0 = r00 > 0 ? a0 / r00 : Vector3<T>{};
    const auto r01 = dot( e0, a1 );
    const auto r02 = dot( e0, a2 );
    a1 -= r01 * e0;
    const auto r11 = a1.length();
    const auto e1 = r11 > 0 ? a1 / r11 : Vector3<T>{};
    const auto r12 = dot( e1, a2 );
    a2 -= r02 * e0 + r12 * e1;
    const auto r22 = a2.length();
    const auto e2 = r22 > 0 ? a2 / r22 : Vector3<T>{};
    return QR
    {
        Matrix3::fromColumns( e0, e1, e2 ),
        Matrix3::fromRows( { r00, r01, r02 }, { T(0), r11, r12 }, { T(0), T(0), r22 } )
    };
}

/// \}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace MR
