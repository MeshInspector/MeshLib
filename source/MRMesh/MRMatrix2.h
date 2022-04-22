#pragma once

#include "MRVector2.h"
#include "MRConstants.h"

namespace MR
{

// arbitrary 2x2 matrix
template <typename T>
struct Matrix2
{
    using ValueType = T;
    using VectorType = Vector2<T>;

    // rows, identity matrix by default
    Vector2<T> x{ 1, 0 };
    Vector2<T> y{ 0, 1 };

    constexpr Matrix2() noexcept = default;
    // initializes matrix from its 2 rows
    constexpr Matrix2( const Vector2<T> & x, const Vector2<T> & y ) : x( x ), y( y ) { }
    template <typename U>
    constexpr explicit Matrix2( const Matrix2<U> & m ) : x( m.x ), y( m.y ) { }
    static constexpr Matrix2 zero() noexcept { return Matrix2( Vector2<T>(), Vector2<T>() ); }
    // returns a matrix that scales uniformly
    static constexpr Matrix2 scale( T s ) noexcept { return Matrix2( { s, T(0) }, { T(0), s } ); }
    // returns a matrix that has its own scale along each axis
    static constexpr Matrix2 scale( T sx, T sy ) noexcept { return Matrix2( { sx, T(0) }, { T(0), sy } ); }
    static constexpr Matrix2 scale( const Vector2<T> & s ) noexcept { return Matrix2( { s.x, T(0) }, { T(0), s.y } ); }
    // creates matrix representing rotation around origin on given angle
    static constexpr Matrix2 rotation( T angle ) noexcept;
    // creates matrix representing rotation that after application to (from) makes (to) vector
    static constexpr Matrix2 rotation( const Vector2<T> & from, const Vector2<T> & to ) noexcept;
    // constructs a matrix from its 2 rows
    static constexpr Matrix2 fromRows( const Vector2<T> & x, const Vector2<T> & y ) noexcept { return Matrix2( x, y ); }
    // constructs a matrix from its 2 columns;
    // use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
    static constexpr Matrix2 fromColumns( const Vector2<T> & x, const Vector2<T> & y ) noexcept { return Matrix2( x, y ).transposed(); }

    // row access
    constexpr const Vector2<T> & operator []( int row ) const noexcept { return *( &x + row ); }
    constexpr       Vector2<T> & operator []( int row )       noexcept { return *( &x + row ); }

    // column access
    constexpr Vector2<T> col( int i ) const noexcept { return { x[i], y[i] }; }

    // computes trace of the matrix
    constexpr T trace() const noexcept { return x.x + y.y; }
    // compute sum of squared matrix elements
    constexpr T normSq() const noexcept { return x.lengthSq() + y.lengthSq(); }
    constexpr T norm() const noexcept { return std::sqrt( normSq() ); }
    // computes determinant of the matrix
    constexpr T det() const noexcept;
    // computes inverse matrix
    constexpr Matrix2<T> inverse() const noexcept;
    // computes transposed matrix
    constexpr Matrix2<T> transposed() const noexcept;

    Matrix2 & operator +=( const Matrix2<T> & b ) { x += b.x; y += b.y; return * this; }
    Matrix2 & operator -=( const Matrix2<T> & b ) { x -= b.x; y -= b.y; return * this; }
    Matrix2 & operator *=( T b ) { x *= b; y *= b; return * this; }
    Matrix2 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { x /= b; y /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }
};

// x = a * b
template <typename T>
inline Vector2<T> operator *( const Matrix2<T> & a, const Vector2<T> & b )
{
    return { dot( a.x, b ), dot( a.y, b ) };
}

// product of two matrices
template <typename T>
inline Matrix2<T> operator *( const Matrix2<T> & a, const Matrix2<T> & b )
{
    Matrix2<T> res;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 2; ++j )
            res[i][j] = dot( a[i], b.col(j) );
    return res;
}

// x = a * b^T
template <typename T>
inline Matrix2<T> outer( const Vector2<T> & a, const Vector2<T> & b )
{
    return { a.x * b, a.y * b };
}

template <typename T>
inline bool operator ==( const Matrix2<T> & a, const Matrix2<T> & b )
    { return a.x == b.x && a.y == b.y; }

template <typename T>
inline bool operator !=( const Matrix2<T> & a, const Matrix2<T> & b )
    { return !( a == b ); }

template <typename T>
inline Matrix2<T> operator +( const Matrix2<T> & a, const Matrix2<T> & b )
    { return { a.x + b.x, a.y + b.y }; }

template <typename T>
inline Matrix2<T> operator -( const Matrix2<T> & a, const Matrix2<T> & b )
    { return { a.x - b.x, a.y - b.y }; }

template <typename T>
inline Matrix2<T> operator *( T a, const Matrix2<T> & b )
    { return { a * b.x, a * b.y }; }

template <typename T>
inline Matrix2<T> operator *( const Matrix2<T> & b, T a )
    { return { a * b.x, a * b.y }; }

template <typename T>
inline Matrix2<T> operator /( Matrix2<T> b, T a )
    { b /= a; return b; }

template <typename T>
constexpr Matrix2<T> Matrix2<T>::rotation( T angle ) noexcept
{
    T c = cos( angle );
    T s = sin( angle );
    return {
        { c, -s },
        { s,  c }
    };
}

template <typename T>
constexpr Matrix2<T> Matrix2<T>::rotation( const Vector2<T> & from, const Vector2<T> & to ) noexcept
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
constexpr Matrix2<T> Matrix2<T>::inverse() const noexcept
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

} //namespace MR
