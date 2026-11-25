#pragma once

#include "MRVector4.h"
#include "MRMatrix4.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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

    /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
    /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
    /// M*V^T = V^T*L; M = V^T*L*V
    Vector4<T> eigens( Matrix4<T> * eigenvectors = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );

    /// for not-degenerate matrix returns just inverse matrix, otherwise
    /// returns degenerate matrix, which performs inversion on not-kernel subspace;
    /// \param tol relative epsilon-tolerance for too small number detection
    /// \param rank optional output for this matrix rank according to given tolerance
    SymMatrix4<T> pseudoinverse( T tol = std::numeric_limits<T>::epsilon(), int * rank = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );

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
                                        ww += b.ww;
    return * this;
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator -=( const SymMatrix4<T> & b )
{
    xx -= b.xx; xy -= b.xy; xz -= b.xz; xw -= b.xw;
                yy -= b.yy; yz -= b.yz; yw -= b.yw;
                            zz -= b.zz; zw -= b.zw;
                                        ww -= b.ww;
    return * this;
}

template <typename T>
SymMatrix4<T> & SymMatrix4<T>::operator *=( T b )
{
    xx *= b; xy *= b; xz *= b; xw *= b;
             yy *= b; yz *= b; yw *= b;
                      zz *= b; zw *= b;
                               ww *= b;
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
                                   ww /= b;
    }
    else
        *this *= ( 1 / b );
    return * this;
}

template <typename T>
Vector4<T> SymMatrix4<T>::eigens( Matrix4<T> * eigenvectors ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // cyclic Jacobi method: each rotation zeroes one off-diagonal element,
    // it is precise also for close and for tiny eigenvalues
    T a[4][4] =
    {
        { xx, xy, xz, xw },
        { xy, yy, yz, yw },
        { xz, yz, zz, zw },
        { xw, yw, zw, ww }
    };
    Matrix4<T> v; // eigenvectors in rows
    for ( int sweep = 0; sweep < 32; ++sweep )
    {
        bool rotated = false;
        for ( int p = 0; p < 3; ++p )
            for ( int q = p + 1; q < 4; ++q )
            {
                const T apq = a[p][q];
                // the rotation is skipped if it cannot change the diagonal elements
                if ( apq == 0 || std::abs( apq ) <= std::numeric_limits<T>::epsilon() * std::sqrt( std::abs( a[p][p] * a[q][q] ) ) / 8 )
                {
                    a[p][q] = a[q][p] = 0;
                    continue;
                }
                rotated = true;
                const T theta = ( a[q][q] - a[p][p] ) / ( 2 * apq );
                const T t = ( theta >= 0 ? T( 1 ) : T( -1 ) ) / ( std::abs( theta ) + std::sqrt( theta * theta + 1 ) );
                const T c = 1 / std::sqrt( t * t + 1 );
                const T s = t * c;
                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = a[q][p] = 0;
                for ( int k = 0; k < 4; ++k )
                {
                    if ( k != p && k != q )
                    {
                        const T akp = a[k][p], akq = a[k][q];
                        a[k][p] = a[p][k] = c * akp - s * akq;
                        a[k][q] = a[q][k] = s * akp + c * akq;
                    }
                    const T vpk = v[p][k], vqk = v[q][k];
                    v[p][k] = c * vpk - s * vqk;
                    v[q][k] = s * vpk + c * vqk;
                }
            }
        if ( !rotated )
            break;
    }
    int order[4] = { 0, 1, 2, 3 };
    std::sort( order, order + 4, [&]( int i, int j ) { return a[i][i] < a[j][j]; } );
    Vector4<T> eig;
    for ( int i = 0; i < 4; ++i )
        eig[i] = a[order[i]][order[i]];
    if ( eigenvectors )
        *eigenvectors = Matrix4<T>( v[order[0]], v[order[1]], v[order[2]], v[order[3]] );
    return eig;
}

template <typename T>
SymMatrix4<T> SymMatrix4<T>::pseudoinverse( T tol, int * rank ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    SymMatrix4<T> res;
    Matrix4<T> eigenvectors;
    const auto eigenvalues = eigens( &eigenvectors );
    const auto threshold = std::max( std::abs( eigenvalues[0] ), std::abs( eigenvalues[3] ) ) * tol;
    int myRank = 0;
    for ( int i = 0; i < 4; ++i )
    {
        if ( std::abs( eigenvalues[i] ) <= threshold )
            continue;
        res += outerSquare( 1 / eigenvalues[i], eigenvectors[i] );
        ++myRank;
    }
    if ( rank )
        *rank = myRank;
    return res;
}

/// \}

} // namespace MR
