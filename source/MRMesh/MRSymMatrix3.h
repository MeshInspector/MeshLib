#pragma once

#include "MRVector3.h"
#include "MRMatrix3.h"
#include "MRConstants.h"
#include <limits>

namespace MR
{
 
/// symmetric 3x3 matrix
/// \ingroup MatrixGroup
template <typename T>
struct SymMatrix3
{
    using ValueType = T;

    /// zero matrix by default
    T xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    constexpr SymMatrix3() noexcept = default;
    template <typename U>
    constexpr explicit SymMatrix3( const SymMatrix3<U> & m ) : xx( T( m.xx ) ), xy( T( m.xy ) ), xz( T( m.xz ) ), yy( T( m.yy ) ), yz( T( m.yz ) ), zz( T( m.zz ) ) { }
    static constexpr SymMatrix3 identity() noexcept { SymMatrix3 res; res.xx = res.yy = res.zz = 1; return res; }
    static constexpr SymMatrix3 diagonal( T diagVal ) noexcept { SymMatrix3 res; res.xx = res.yy = res.zz = diagVal; return res; }

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return xx + yy + zz; }
    /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
    constexpr T normSq() const noexcept;
    /// computes determinant of the matrix
    constexpr T det() const noexcept;
    /// computes inverse matrix
    constexpr SymMatrix3<T> inverse() const noexcept;

    SymMatrix3 & operator +=( const SymMatrix3<T> & b ) { xx += b.xx; xy += b.xy; xz += b.xz; yy += b.yy; yz += b.yz; zz += b.zz; return * this; }
    SymMatrix3 & operator -=( const SymMatrix3<T> & b ) { xx -= b.xx; xy -= b.xy; xz -= b.xz; yy -= b.yy; yz -= b.yz; zz -= b.zz; return * this; }
    SymMatrix3 & operator *=( T b ) { xx *= b; xy *= b; xz *= b; yy *= b; yz *= b; zz *= b; return * this; }
    SymMatrix3 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { xx /= b; xy /= b; xz /= b; yy /= b; yz /= b; zz /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }

    /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
    /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
    /// M*V^T = V^T*L; M = V^T*L*V
    Vector3<T> eigens( Matrix3<T> * eigenvectors = nullptr ) const;
    /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
    Vector3<T> eigenvector( T eigenvalue ) const;
};

/// \related SymMatrix3
/// \{

/// x = a * b
template <typename T> 
inline Vector3<T> operator *( const SymMatrix3<T> & a, const Vector3<T> & b )
{
    return 
    { 
        a.xx * b.x + a.xy * b.y + a.xz * b.z,
        a.xy * b.x + a.yy * b.y + a.yz * b.z,
        a.xz * b.x + a.yz * b.y + a.zz * b.z
    };
}

/// x = a * a^T
template <typename T> 
inline SymMatrix3<T> outerSquare( const Vector3<T> & a )
{
    SymMatrix3<T> res;
    res.xx = a.x * a.x;
    res.xy = a.x * a.y;
    res.xz = a.x * a.z;
    res.yy = a.y * a.y;
    res.yz = a.y * a.z;
    res.zz = a.z * a.z;
    return res;
}

template <typename T> 
inline bool operator ==( const SymMatrix3<T> & a, const SymMatrix3<T> & b )
    { return a.xx = b.xx && a.xy = b.xy && a.xz = b.xz && a.yy = b.yy && a.yz = b.yz && a.zz = b.zz; }

template <typename T> 
inline bool operator !=( const SymMatrix3<T> & a, const SymMatrix3<T> & b )
    { return !( a == b ); }

template <typename T> 
inline SymMatrix3<T> operator +( const SymMatrix3<T> & a, const SymMatrix3<T> & b )
    { SymMatrix3<T> res{ a }; res += b; return res; }

template <typename T> 
inline SymMatrix3<T> operator -( const SymMatrix3<T> & a, const SymMatrix3<T> & b )
    { SymMatrix3<T> res{ a }; res -= b; return res; }

template <typename T> 
inline SymMatrix3<T> operator *( T a, const SymMatrix3<T> & b )
    { SymMatrix3<T> res{ b }; res *= a; return res; }

template <typename T> 
inline SymMatrix3<T> operator *( const SymMatrix3<T> & b, T a )
    { SymMatrix3<T> res{ b }; res *= a; return res; }

template <typename T> 
inline SymMatrix3<T> operator /( SymMatrix3<T> b, T a )
    { b /= a; return b; }

template <typename T> 
constexpr T SymMatrix3<T>::det() const noexcept
{
    return
        xx * ( yy * zz - yz * yz )
     -  xy * ( xy * zz - yz * xz )
     +  xz * ( xy * yz - yy * xz );
}

template <typename T> 
constexpr T SymMatrix3<T>::normSq() const noexcept
{
    return sqr( xx ) + sqr( yy ) + sqr( zz ) +
        2 * ( sqr( xy ) + sqr( xz ) + sqr( yz ) );
}

template <typename T> 
constexpr SymMatrix3<T> SymMatrix3<T>::inverse() const noexcept
{
    auto det = this->det();
    if ( det == 0 )
        return {};
    SymMatrix3<T> res;
    res.xx = ( yy * zz - yz * yz ) / det;
    res.xy = ( xz * yz - xy * zz ) / det;
    res.xz = ( xy * yz - xz * yy ) / det;
    res.yy = ( xx * zz - xz * xz ) / det;
    res.yz = ( xz * xy - xx * yz ) / det;
    res.zz = ( xx * yy - xy * xy ) / det;
    return res;
}

template <typename T> 
Vector3<T> SymMatrix3<T>::eigens( Matrix3<T> * eigenvectors ) const
{
    //https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    const auto q = trace() / 3;
    const auto B = *this - diagonal( q );
    const auto p2 = B.normSq();
    const auto p = std::sqrt( p2 / 6 );
    Vector3<T> eig;
    if ( p <= std::abs( q ) * std::numeric_limits<T>::epsilon() )
    {
        // this is proportional to identity matrix
        eig = { q, q, q };
        if ( eigenvectors )
            *eigenvectors = Matrix3<T>{};
        return eig;
    }
    const auto r = B.det() / ( 2 * p * p * p );

    // In exact arithmetic for a symmetric matrix - 1 <= r <= 1
    // but computation error can leave it slightly outside this range.
    if ( r <= -1 )
    {
        //phi = PI / 3;
        eig[0] = q - 2 * p;
        eig[1] = eig[2] = q + p;
        if ( eigenvectors )
        {
            const auto x = eigenvector( eig[0] ).normalized();
            const auto [ y, z ] = x.perpendicular();
            *eigenvectors = Matrix3<T>::fromRows( x, y, z );
        }
        return eig;
    }
    if ( r >= 1 )
    {
        //phi = 0;
        eig[0] = eig[1] = q - p;
        eig[2] = q + 2 * p;
        if ( eigenvectors )
        {
            const auto z = eigenvector( eig[2] ).normalized();
            const auto [ x, y ] = z.perpendicular();
            *eigenvectors = Matrix3<T>::fromRows( x, y, z );
        }
        return eig;
    }
    const auto phi = std::acos( r ) / 3;
    eig[0] = q + 2 * p * cos( phi + T( 2 * PI / 3 ) );
    eig[2] = q + 2 * p * cos( phi );
    eig[1] = 3 * q - eig[0] - eig[2]; // 2 * q = trace() = eig[0] + eig[1] + eig[2]
    if ( eigenvectors )
    {
        const auto x = eigenvector( eig[0] ).normalized();
        const auto z = eigenvector( eig[2] ).normalized();
        const auto y = cross( z, x );
        *eigenvectors = Matrix3<T>::fromRows( x, y, z );
    }
    return eig;
}

template <typename T> 
Vector3<T> SymMatrix3<T>::eigenvector( T eigenvalue ) const
{
    const Vector3<T> row0( xx - eigenvalue, xy, xz );
    const Vector3<T> row1( xy, yy - eigenvalue, yz );
    const Vector3<T> row2( xz, yz, zz - eigenvalue );
    // not-repeating eigenvalue means that some two rows are linearly independent
    const Vector3<T> crs01 = cross( row0, row1 );
    const Vector3<T> crs12 = cross( row1, row2 );
    const Vector3<T> crs20 = cross( row2, row0 );
    const T lsq01 = crs01.lengthSq();
    const T lsq12 = crs12.lengthSq();
    const T lsq20 = crs20.lengthSq();
    if ( lsq01 > lsq12 )
    {
        if ( lsq01 > lsq20 )
            return crs01;
    }
    else if ( lsq12 > lsq20 )
        return crs12;
    return crs20;
}

/// \}

} // namespace MR
