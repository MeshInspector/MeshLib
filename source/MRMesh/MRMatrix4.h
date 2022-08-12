#pragma once

#include "MRVector4.h"
#include <limits>
#include <cassert>

namespace MR
{

/// arbitrary 4x4 matrix
/// \ingroup MatrixGroup
template <typename T>
struct Matrix4
{
    using ValueType = T;
    using VectorType = Vector4<T>;

    /// rows, identity matrix by default
    Vector4<T> x{ 1, 0, 0, 0 };
    Vector4<T> y{ 0, 1, 0, 0 };
    Vector4<T> z{ 0, 0, 1, 0 };
    Vector4<T> w{ 0, 0, 0, 1 };

    constexpr Matrix4() noexcept = default;
    /// initializes matrix from 4 row-vectors
    constexpr Matrix4( const Vector4<T>& x, const Vector4<T>& y, const Vector4<T>& z, const Vector4<T>& w ) : x( x ), y( y ), z( z ), w( w ) { }

    /// construct from rotation matrix and translation vector
    constexpr Matrix4( const Matrix3<T>& r, const Vector3<T>& t )
    {
        x = Vector4<T>( r.x.x, r.x.y, r.x.z, t.x );
        y = Vector4<T>( r.y.x, r.y.y, r.y.z, t.y );
        z = Vector4<T>( r.z.x, r.z.y, r.z.z, t.z );
        w = Vector4<T>( 0, 0, 0, 1 );
    }

    constexpr Matrix4( const AffineXf3<T>& xf ) : Matrix4( xf.A, xf.b ) {}

    template <typename U>
    constexpr explicit Matrix4( const Matrix4<U> & m ) : x( m.x ), y( m.y ), z( m.z ), w( m.w ) { }
    static constexpr Matrix4 zero() noexcept { return Matrix4( Vector4<T>(), Vector4<T>(), Vector4<T>(), Vector4<T>() ); }
    /// returns a matrix that scales uniformly
    static constexpr Matrix4 scale( T s ) noexcept { return Matrix4( { s, T(0), T(0), T(0) }, { T(0), s, T(0), T(0) }, { T(0), T(0), s, T(0) }, { T(0), T(0), T(0), s } ); }

    /// element access
    constexpr const T& operator ()( int row, int col ) const noexcept { return operator[]( row )[col]; }
    constexpr       T& operator ()( int row, int col )       noexcept { return operator[]( row )[col]; }

    /// row access
    constexpr const Vector4<T> & operator []( int row ) const noexcept { return *( &x + row ); }
    constexpr       Vector4<T> & operator []( int row )       noexcept { return *( &x + row ); }

    /// column access
    constexpr Vector4<T> col( int i ) const noexcept { return { x[i], y[i], z[i], w[i] }; }

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return x.x + y.y + z.z + w.w; }
    /// compute sum of squared matrix elements
    constexpr T normSq() const noexcept { return x.lengthSq() + y.lengthSq() + z.lengthSq() + w.lengthSq(); }
    constexpr T norm() const noexcept { return std::sqrt( normSq() ); }
    /// computes submatrix of the matrix with excluded i-th row and j-th column
    Matrix3<T> submatrix( int i, int j ) const noexcept;
    /// computes determinant of the matrix
    T det() const noexcept;
    /// computes inverse matrix
    constexpr Matrix4<T> inverse() const noexcept;
    /// computes transposed matrix
    constexpr Matrix4<T> transposed() const noexcept;

    constexpr Matrix3<T> getRotation() const noexcept;
    void setRotation( const Matrix3<T>& rot) noexcept;
    constexpr Vector3<T> getTranslation() const noexcept;
    void setTranslation( const Vector3<T>& t ) noexcept;

    constexpr T* data() { return (T*) (&x); };
    Matrix4 & operator +=( const Matrix4<T> & b ) { x += b.x; y += b.y; z += b.z; w += b.w; return * this; }
    Matrix4 & operator -=( const Matrix4<T> & b ) { x -= b.x; y -= b.y; z -= b.z; w -= b.w; return * this; }
    Matrix4 & operator *=( T b ) { x *= b; y *= b; z *= b; w *= b; return * this; }
    Matrix4 & operator /=( T b )
    {
        if constexpr ( std::is_integral_v<T> )
            { x /= b; y /= b; z /= b; w /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }

    operator AffineXf3<T>() const
    {
        assert( std::fabs( w.x )     < std::numeric_limits<T>::epsilon() * 1000 );
        assert( std::fabs( w.y )     < std::numeric_limits<T>::epsilon() * 1000 );
        assert( std::fabs( w.z )     < std::numeric_limits<T>::epsilon() * 1000 );
        assert( std::fabs( 1 - w.w ) < std::numeric_limits<T>::epsilon() * 1000 );
        AffineXf3<T> res;
        res.A.x.x = x.x; res.A.x.y = x.y; res.A.x.z = x.z; res.b.x = x.w;
        res.A.y.x = y.x; res.A.y.y = y.y; res.A.y.z = y.z; res.b.y = y.w;
        res.A.z.x = z.x; res.A.z.y = z.y; res.A.z.z = z.z; res.b.z = z.w;
        return res;
    }

    /// converts 3d-vector b in 4d-vector (b,1), multiplies matrix on it, 
    /// and assuming the result is in homogeneous coordinates returns it as 3d-vector
    Vector3<T> operator ()( const Vector3<T> & b ) const;
};

/// \related Matrix4
/// \{

/// x = a * b
template <typename T>
inline Vector4<T> operator *( const Matrix4<T> & a, const Vector4<T> & b )
{
    return { dot( a.x, b ), dot( a.y, b ), dot( a.z, b ), dot( a.w, b ) };
}

template <typename T>
inline Vector3<T> Matrix4<T>::operator ()( const Vector3<T> & b ) const
{
    return ( *this * Vector4<T>{ b.x, b.y, b.z, T(1) } ).proj3d();
}

/// product of two matrices
template <typename T>
inline Matrix4<T> operator *( const Matrix4<T> & a, const Matrix4<T> & b )
{
    Matrix4<T> res;
    for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 4; ++j )
            res[i][j] = dot( a[i], b.col(j) );
    return res;
}

/// x = a * b^T
template <typename T>
inline Matrix4<T> outer( const Vector4<T> & a, const Vector4<T> & b )
{
    return { a.x * b, a.y * b, a.z * b, a.w * b };
}

template <typename T>
inline bool operator ==( const Matrix4<T> & a, const Matrix4<T> & b )
    { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

template <typename T>
inline bool operator !=( const Matrix4<T> & a, const Matrix4<T> & b )
    { return !( a == b ); }

template <typename T>
inline Matrix4<T> operator +( const Matrix4<T> & a, const Matrix4<T> & b )
    { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }

template <typename T>
inline Matrix4<T> operator -( const Matrix4<T> & a, const Matrix4<T> & b )
    { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }

template <typename T>
inline Matrix4<T> operator *( T a, const Matrix4<T> & b )
    { return { a * b.x, a * b.y, a * b.z, a * b.w }; }

template <typename T>
inline Matrix4<T> operator *( const Matrix4<T> & b, T a )
    { return { a * b.x, a * b.y, a * b.z, a * b.z }; }

template <typename T>
inline Matrix4<T> operator /( Matrix4<T> b, T a )
    { b /= a; return b; }

template <typename T>
Matrix3<T> Matrix4<T>::submatrix( int i, int j ) const noexcept
{
    Matrix3<T> res;
    auto* resM = (T*) &res.x;
    int cur = 0;
    for ( int m = 0; m < 4; m++ )
    {
        if ( m == i )
            continue;
        for ( int n = 0; n < 4; n++ )
        {
            if ( n == j )
                continue;
            resM[cur++] = (*this)[m][n];
        }
    }
    assert( cur == 9 );
    return res;
}

template <typename T>
T Matrix4<T>::det() const noexcept
{
    return
        x.x * submatrix( 0, 0 ).det()
      - x.y * submatrix( 0, 1 ).det()
      + x.z * submatrix( 0, 2 ).det()
      - x.w * submatrix( 0, 3 ).det();
}

template <typename T>
constexpr Matrix4<T> Matrix4<T>::transposed() const noexcept
{
    return Matrix4<T>
    {
        { x.x, y.x, z.x, w.x },
        { x.y, y.y, z.y, w.y },
        { x.z, y.z, z.z, w.z },
        { x.w, y.w, z.w, w.w },
    };
}

template <typename T>
constexpr Matrix4<T> Matrix4<T>::inverse() const noexcept
{
    Matrix4<T> inv;
    T* m = (T*) (&x);
    T det;

    inv[0][0] = m[5] * m[10] * m[15] -
        m[5] * m[11] * m[14] -
        m[9] * m[6] * m[15] +
        m[9] * m[7] * m[14] +
        m[13] * m[6] * m[11] -
        m[13] * m[7] * m[10];

    inv[1][0] = -m[4] * m[10] * m[15] +
        m[4] * m[11] * m[14] +
        m[8] * m[6] * m[15] -
        m[8] * m[7] * m[14] -
        m[12] * m[6] * m[11] +
        m[12] * m[7] * m[10];

    inv[2][0] = m[4] * m[9] * m[15] -
        m[4] * m[11] * m[13] -
        m[8] * m[5] * m[15] +
        m[8] * m[7] * m[13] +
        m[12] * m[5] * m[11] -
        m[12] * m[7] * m[9];

    inv[3][0] = -m[4] * m[9] * m[14] +
        m[4] * m[10] * m[13] +
        m[8] * m[5] * m[14] -
        m[8] * m[6] * m[13] -
        m[12] * m[5] * m[10] +
        m[12] * m[6] * m[9];

    inv[0][1] = -m[1] * m[10] * m[15] +
        m[1] * m[11] * m[14] +
        m[9] * m[2] * m[15] -
        m[9] * m[3] * m[14] -
        m[13] * m[2] * m[11] +
        m[13] * m[3] * m[10];

    inv[1][1] = m[0] * m[10] * m[15] -
        m[0] * m[11] * m[14] -
        m[8] * m[2] * m[15] +
        m[8] * m[3] * m[14] +
        m[12] * m[2] * m[11] -
        m[12] * m[3] * m[10];

    inv[2][1] = -m[0] * m[9] * m[15] +
        m[0] * m[11] * m[13] +
        m[8] * m[1] * m[15] -
        m[8] * m[3] * m[13] -
        m[12] * m[1] * m[11] +
        m[12] * m[3] * m[9];

    inv[3][1] = m[0] * m[9] * m[14] -
        m[0] * m[10] * m[13] -
        m[8] * m[1] * m[14] +
        m[8] * m[2] * m[13] +
        m[12] * m[1] * m[10] -
        m[12] * m[2] * m[9];

    inv[0][2] = m[1] * m[6] * m[15] -
        m[1] * m[7] * m[14] -
        m[5] * m[2] * m[15] +
        m[5] * m[3] * m[14] +
        m[13] * m[2] * m[7] -
        m[13] * m[3] * m[6];

    inv[1][2] = -m[0] * m[6] * m[15] +
        m[0] * m[7] * m[14] +
        m[4] * m[2] * m[15] -
        m[4] * m[3] * m[14] -
        m[12] * m[2] * m[7] +
        m[12] * m[3] * m[6];

    inv[2][2] = m[0] * m[5] * m[15] -
        m[0] * m[7] * m[13] -
        m[4] * m[1] * m[15] +
        m[4] * m[3] * m[13] +
        m[12] * m[1] * m[7] -
        m[12] * m[3] * m[5];

    inv[3][2] = -m[0] * m[5] * m[14] +
        m[0] * m[6] * m[13] +
        m[4] * m[1] * m[14] -
        m[4] * m[2] * m[13] -
        m[12] * m[1] * m[6] +
        m[12] * m[2] * m[5];

    inv[0][3] = -m[1] * m[6] * m[11] +
        m[1] * m[7] * m[10] +
        m[5] * m[2] * m[11] -
        m[5] * m[3] * m[10] -
        m[9] * m[2] * m[7] +
        m[9] * m[3] * m[6];

    inv[1][3] = m[0] * m[6] * m[11] -
        m[0] * m[7] * m[10] -
        m[4] * m[2] * m[11] +
        m[4] * m[3] * m[10] +
        m[8] * m[2] * m[7] -
        m[8] * m[3] * m[6];

    inv[2][3] = -m[0] * m[5] * m[11] +
        m[0] * m[7] * m[9] +
        m[4] * m[1] * m[11] -
        m[4] * m[3] * m[9] -
        m[8] * m[1] * m[7] +
        m[8] * m[3] * m[5];

    inv[3][3] = m[0] * m[5] * m[10] -
        m[0] * m[6] * m[9] -
        m[4] * m[1] * m[10] +
        m[4] * m[2] * m[9] +
        m[8] * m[1] * m[6] -
        m[8] * m[2] * m[5];

    det = m[0] * inv[0][0] + m[1] * inv[1][0] + m[2] * inv[2][0] + m[3] * inv[3][0];

    if( det == 0 )
    {
        // impossible to invert singular matrix
        assert( false );
        return Matrix4<T>();
    }

    inv /= det;

    return inv;
}

template <typename T>
constexpr Matrix3<T> Matrix4<T>::getRotation() const noexcept
{
    return Matrix3<T>
    {
        { x.x, x.y, x.z },
        { y.x, y.y, y.z },
        { z.x, z.y, z.z }
    };
}

template <typename T>
void Matrix4<T>::setRotation( const Matrix3<T>& rot ) noexcept
{
    x.x = rot.x.x; x.y = rot.x.y; x.z = rot.x.z;
    y.x = rot.y.x; y.y = rot.y.y; y.z = rot.y.z;
    z.x = rot.z.x; z.y = rot.z.y; z.z = rot.z.z;
}

template <typename T>
constexpr Vector3<T> Matrix4<T>::getTranslation() const noexcept
{
    return Vector3<T>{ x.w, y.w, z.w };
}

template <typename T>
void Matrix4<T>::setTranslation( const Vector3<T>& t ) noexcept
{
    x.w = t.x; y.w = t.y; z.w = t.z;
}

/// \}

} // namespace MR
