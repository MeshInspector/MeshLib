#pragma once

#include "MRVector3.h"
#include "MRConstants.h"

namespace MR
{

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
    /// returns a matrix that scales uniformly
    static constexpr Matrix3 scale( T s ) noexcept { return Matrix3( { s, T(0), T(0) }, { T(0), s, T(0) }, { T(0), T(0), s } ); }
    /// returns a matrix that has its own scale along each axis
    static constexpr Matrix3 scale( T sx, T sy, T sz ) noexcept { return Matrix3( { sx, T(0), T(0) }, { T(0), sy, T(0) }, { T(0), T(0), sz } ); }
    static constexpr Matrix3 scale( const Vector3<T> & s ) noexcept { return Matrix3( { s.x, T(0), T(0) }, { T(0), s.y, T(0) }, { T(0), T(0), s.z } ); }
    /// creates matrix representing rotation around given axis on given angle
    static constexpr Matrix3 rotation( const Vector3<T> & axis, T angle ) noexcept;
    /// creates matrix representing rotation that after application to (from) makes (to) vector
    static constexpr Matrix3 rotation( const Vector3<T> & from, const Vector3<T> & to ) noexcept;
    /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
    /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
    static constexpr Matrix3 rotationFromEuler( const Vector3<T> & eulerAngles ) noexcept;
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
    constexpr T norm() const noexcept { return std::sqrt( normSq() ); }
    /// computes determinant of the matrix
    constexpr T det() const noexcept;
    /// computes inverse matrix
    constexpr Matrix3<T> inverse() const noexcept;
    /// computes transposed matrix
    constexpr Matrix3<T> transposed() const noexcept;
    /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
    constexpr Vector3<T> toEulerAngles() const noexcept;
    /// returns scaling factors by axes (Ox, Oy, Oz)
    constexpr Vector3<T> toScale() const noexcept;

    Matrix3 & operator +=( const Matrix3<T> & b ) { x += b.x; y += b.y; z += b.z; return * this; }
    Matrix3 & operator -=( const Matrix3<T> & b ) { x -= b.x; y -= b.y; z -= b.z; return * this; }
    Matrix3 & operator *=( T b ) { x *= b; y *= b; z *= b; return * this; }
    Matrix3 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { x /= b; y /= b; z /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }
};

/// \related Matrix3
/// \{

/// x = a * b
template <typename T>
inline Vector3<T> operator *( const Matrix3<T> & a, const Vector3<T> & b )
{
    return { dot( a.x, b ), dot( a.y, b ), dot( a.z, b ) };
}

/// product of two matrices
template <typename T>
inline Matrix3<T> operator *( const Matrix3<T> & a, const Matrix3<T> & b )
{
    Matrix3<T> res;
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            res[i][j] = dot( a[i], b.col(j) );
    return res;
}

/// x = a * b^T
template <typename T>
inline Matrix3<T> outer( const Vector3<T> & a, const Vector3<T> & b )
{
    return { a.x * b, a.y * b, a.z * b };
}

template <typename T>
inline bool operator ==( const Matrix3<T> & a, const Matrix3<T> & b )
    { return a.x == b.x && a.y == b.y && a.z == b.z; }

template <typename T>
inline bool operator !=( const Matrix3<T> & a, const Matrix3<T> & b )
    { return !( a == b ); }

template <typename T>
inline Matrix3<T> operator +( const Matrix3<T> & a, const Matrix3<T> & b )
    { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

template <typename T>
inline Matrix3<T> operator -( const Matrix3<T> & a, const Matrix3<T> & b )
    { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

template <typename T>
inline Matrix3<T> operator *( T a, const Matrix3<T> & b )
    { return { a * b.x, a * b.y, a * b.z }; }

template <typename T>
inline Matrix3<T> operator *( const Matrix3<T> & b, T a )
    { return { a * b.x, a * b.y, a * b.z }; }

template <typename T>
inline Matrix3<T> operator /( Matrix3<T> b, T a )
    { b /= a; return b; }

template <typename T>
constexpr Matrix3<T> Matrix3<T>::rotation( const Vector3<T> & axis, T angle ) noexcept
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
constexpr Matrix3<T> Matrix3<T>::rotation( const Vector3<T> & from, const Vector3<T> & to ) noexcept
{
    auto axis = cross( from, to );
    if ( axis.lengthSq() > 0 )
        return rotation( axis, angle( from, to ) );
    if ( dot( from, to ) >= 0 )
        return {}; // identity matrix
    return rotation( cross( from, from.furthestBasisVector() ), T( PI ) );
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::rotationFromEuler( const Vector3<T> & eulerAngles ) noexcept
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
constexpr T Matrix3<T>::det() const noexcept
{
    return
        x.x * ( y.y * z.z - y.z * z.y )
     -  x.y * ( y.x * z.z - y.z * z.x )
     +  x.z * ( y.x * z.y - y.y * z.x );
}

template <typename T>
constexpr Matrix3<T> Matrix3<T>::inverse() const noexcept
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
constexpr Vector3<T> Matrix3<T>::toEulerAngles() const noexcept
{
    // https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    return {
        std::atan2(  z.y, z.z ),
        std::atan2( -z.x, std::sqrt( z.y * z.y + z.z * z.z ) ),
        std::atan2(  y.x, x.x )
    };
}

template <typename T>
constexpr Vector3<T> Matrix3<T>::toScale() const noexcept
{
    T scaleX = x.length();
    T scaleY = y.length();
    T scaleZ = z.length();
    return { scaleX, scaleY, scaleZ };
}

/// \}

} // namespace MR
