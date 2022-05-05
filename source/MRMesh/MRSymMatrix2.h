#pragma once

#include "MRVector2.h"

namespace MR
{
 
/// symmetric 2x2 matrix
/// \ingroup MatrixGroup
template <typename T>
struct SymMatrix2
{
    using ValueType = T;

    /// zero matrix by default
    T xx = 0, xy = 0, yy = 0;

    constexpr SymMatrix2() noexcept = default;
    template <typename U>
    constexpr explicit SymMatrix2( const SymMatrix2<U> & m ) : xx( T( m.xx ) ), xy( T( m.xy ) ), yy( T( m.yy ) ) { }
    static constexpr SymMatrix2 identity() noexcept { SymMatrix2 res; res.xx = res.yy = 1; return res; }
    static constexpr SymMatrix2 diagonal( T diagVal ) noexcept { SymMatrix2 res; res.xx = res.yy = diagVal; return res; }

    /// computes determinant of the matrix
    constexpr T det() const noexcept;
    /// computes inverse matrix
    constexpr SymMatrix2<T> inverse() const noexcept;

    SymMatrix2 & operator +=( const SymMatrix2<T> & b ) { xx += b.xx; xy += b.xy; yy += b.yy; return * this; }
    SymMatrix2 & operator -=( const SymMatrix2<T> & b ) { xx -= b.xx; xy -= b.xy; yy -= b.yy; return * this; }
    SymMatrix2 & operator *=( T b ) { xx *= b; xy *= b; yy *= b; return * this; }
    SymMatrix2 & operator /=( T b ) 
    {
        if constexpr ( std::is_integral_v<T> )
            { xx /= b; xy /= b; yy /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }
};

/// \related SymMatrix2
/// \{

/// x = a * b
template <typename T> 
inline Vector2<T> operator *( const SymMatrix2<T> & a, const Vector2<T> & b )
{
    return 
    { 
        a.xx * b.x + a.xy * b.y,
        a.xy * b.x + a.yy * b.y
    };
}

/// x = a * a^T
template <typename T> 
inline SymMatrix2<T> outerSquare( const Vector2<T> & a )
{
    SymMatrix2<T> res;
    res.xx = a.x * a.x;
    res.xy = a.x * a.y;
    res.yy = a.y * a.y;
    return res;
}

template <typename T> 
inline bool operator ==( const SymMatrix2<T> & a, const SymMatrix2<T> & b )
    { return a.xx = b.xx && a.xy = b.xy && a.yy = b.yy; }

template <typename T> 
inline bool operator !=( const SymMatrix2<T> & a, const SymMatrix2<T> & b )
    { return !( a == b ); }

template <typename T> 
inline SymMatrix2<T> operator +( const SymMatrix2<T> & a, const SymMatrix2<T> & b )
    { SymMatrix2<T> res{ a }; res += b; return res; }

template <typename T> 
inline SymMatrix2<T> operator -( const SymMatrix2<T> & a, const SymMatrix2<T> & b )
    { SymMatrix2<T> res{ a }; res -= b; return res; }

template <typename T> 
inline SymMatrix2<T> operator *( T a, const SymMatrix2<T> & b )
    { SymMatrix2<T> res{ b }; res *= a; return res; }

template <typename T> 
inline SymMatrix2<T> operator *( const SymMatrix2<T> & b, T a )
    { SymMatrix2<T> res{ b }; res *= a; return res; }

template <typename T> 
inline SymMatrix2<T> operator /( SymMatrix2<T> b, T a )
    { b /= a; return b; }

template <typename T> 
constexpr T SymMatrix2<T>::det() const noexcept
{
    return xx * yy - xy * xy;
}

template <typename T> 
constexpr SymMatrix2<T> SymMatrix2<T>::inverse() const noexcept
{
    auto det = this->det();
    if ( det == 0 )
        return {};
    SymMatrix2<T> res;
    res.xx =  yy / det;
    res.xy = -xy / det;
    res.yy =  xx / det;
    return res;
}

/// \}

} // namespace MR
