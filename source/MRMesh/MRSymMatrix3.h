#pragma once

#include "MRVector3.h"

namespace MR
{
 
// symmetric 3x3 matrix
template <typename T>
struct SymMatrix3
{
    using ValueType = T;

    // zero matrix by default
    T xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    constexpr SymMatrix3() noexcept = default;
    template <typename U>
    constexpr explicit SymMatrix3( const SymMatrix3<U> & m ) : xx( T( m.xx ) ), xy( T( m.xy ) ), xz( T( m.xz ) ), yy( T( m.yy ) ), yz( T( m.yz ) ), zz( T( m.zz ) ) { }
    static constexpr SymMatrix3 identity() noexcept { SymMatrix3 res; res.xx = res.yy = res.zz = 1; return res; }
    static constexpr SymMatrix3 diagonal( T diagVal ) noexcept { SymMatrix3 res; res.xx = res.yy = res.zz = diagVal; return res; }

    // computes determinant of the matrix
    constexpr T det() const noexcept;
    // computes inverse matrix
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
};

// x = a * b
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

// x = a * a^T
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

} //namespace MR
