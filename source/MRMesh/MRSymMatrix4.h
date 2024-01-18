#pragma once

#include "MRVector4.h"
#include "MRMatrix4.h"

namespace MR
{
 
/// symmetric 4x4 matrix
/// \ingroup MatrixGroup
template <typename T>
struct SymMatrix4
{
    using ValueType = T;

    /// zero matrix by default
    T xx = 0, xy = 0, xz = 0, xw = 0,
              yy = 0, yz = 0, yw = 0,
                      zz = 0, zw = 0,
                              ww = 0;

    constexpr SymMatrix4() noexcept = default;
    template <typename U>
    constexpr explicit SymMatrix4( const SymMatrix4<U> & m ) :
        xx( T( m.xx ) ), xy( T( m.xy ) ), xz( T( m.xz ) ), xw( T( m.xw ) ),
                         yy( T( m.yy ) ), yz( T( m.yz ) ), yw( T( m.yw ) ),
                                          zz( T( m.zz ) ), zw( T( m.zw ) ),
                                                           ww( T( m.ww ) ) {}
    static constexpr SymMatrix4 identity() noexcept { SymMatrix4 res; res.xx = res.yy = res.zz = res.ww = 1; return res; }
    static constexpr SymMatrix4 diagonal( T diagVal ) noexcept { SymMatrix4 res; res.xx = res.yy = res.zz = res.ww = diagVal; return res; }

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return xx + yy + zz + ww; }
    /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
    constexpr T normSq() const noexcept;

    SymMatrix4 & operator +=( const SymMatrix4<T> & b );
    SymMatrix4 & operator -=( const SymMatrix4<T> & b );
    SymMatrix4 & operator *=( T b );
    SymMatrix4 & operator /=( T b );

    bool operator ==( const SymMatrix4<T> & ) const = default;
};

/// \related SymMatrix4
/// \{

/// x = a * b
template <typename T>
inline Vector4<T> operator *( const SymMatrix4<T> & a, const Vector4<T> & b )
{
    return 
    { 
        a.xx * b.x + a.xy * b.y + a.xz * b.z + a.xw * b.w,
        a.xy * b.x + a.yy * b.y + a.yz * b.z + a.yw * b.w,
        a.xz * b.x + a.yz * b.y + a.zz * b.z + a.zw * b.w,
        a.xw * b.x + a.yw * b.y + a.zw * b.z + a.ww * b.w
    };
}

/// x = a * a^T
template <typename T>
inline SymMatrix4<T> outerSquare( const Vector4<T> & a )
{
    SymMatrix4<T> res;
    res.xx = a.x * a.x; res.xy = a.x * a.y; res.xz = a.x * a.z; res.xw = a.x * a.w;
                        res.yy = a.y * a.y; res.yz = a.y * a.z; res.yw = a.y * a.w;
                                            res.zz = a.z * a.z; res.zw = a.z * a.w;
                                                                res.ww = a.w * a.w;
    return res;
}

/// x = k * a * a^T
template <typename T>
inline SymMatrix4<T> outerSquare( T k, const Vector4<T> & a )
{
    const auto ka = k * a;
    SymMatrix4<T> res;
    res.xx = ka.x * a.x; res.xy = ka.x * a.y; res.xz = ka.x * a.z; res.xw = ka.x * a.w;
                         res.yy = ka.y * a.y; res.yz = ka.y * a.z; res.yw = ka.y * a.w;
                                              res.zz = ka.z * a.z; res.zw = ka.z * a.w;
                                                                   res.ww = ka.w * a.w;
    return res;
}

template <typename T>
inline SymMatrix4<T> operator +( const SymMatrix4<T> & a, const SymMatrix4<T> & b )
    { SymMatrix4<T> res{ a }; res += b; return res; }

template <typename T>
inline SymMatrix4<T> operator -( const SymMatrix4<T> & a, const SymMatrix4<T> & b )
    { SymMatrix4<T> res{ a }; res -= b; return res; }

template <typename T>
inline SymMatrix4<T> operator *( T a, const SymMatrix4<T> & b )
    { SymMatrix4<T> res{ b }; res *= a; return res; }

template <typename T>
inline SymMatrix4<T> operator *( const SymMatrix4<T> & b, T a )
    { SymMatrix4<T> res{ b }; res *= a; return res; }

template <typename T>
inline SymMatrix4<T> operator /( SymMatrix4<T> b, T a )
    { b /= a; return b; }

template <typename T>
constexpr T SymMatrix4<T>::normSq() const noexcept
{
    return sqr( xx ) + sqr( yy ) + sqr( zz ) + sqr( ww ) +
        2 * ( sqr( xy ) + sqr( xz ) + sqr( xw ) + sqr( yz ) + sqr( yw ) + sqr( zw ) );
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator +=( const SymMatrix4<T> & b )
{
    xx += b.xx; xy += b.xy; xz += b.xz; xw += b.xw;
                yy += b.yy; yz += b.yz; yw += b.yw;
                            zz += b.zz; zw += b.zw;
                                        zz += b.zz;
    return * this;
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator -=( const SymMatrix4<T> & b )
{
    xx -= b.xx; xy -= b.xy; xz -= b.xz; xw -= b.xw;
                yy -= b.yy; yz -= b.yz; yw -= b.yw;
                            zz -= b.zz; zw -= b.zw;
                                        zz -= b.zz;
    return * this;
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator *=( T b )
{
    xx *= b; xy *= b; xz *= b; xw *= b;
             yy *= b; yz *= b; yw *= b;
                      zz *= b; zw *= b;
                               zz *= b;
    return * this;
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator /=( T b )
{
    if constexpr ( std::is_integral_v<T> )
    {
        xx /= b; xy /= b; xz /= b; xw /= b;
                 yy /= b; yz /= b; yw /= b;
                          zz /= b; zw /= b;
                                   zz /= b;
    }
    else
        *this *= ( 1 / b );
    return * this;
}

/// \}

} // namespace MR
