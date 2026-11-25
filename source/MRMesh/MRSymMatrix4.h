#pragma once

#include "MRVector4.h"
#include "MRMatrix4.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
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

    /// solves M*x = b for positive definite M by LDL^T decomposition without pivoting,
    /// much faster than eigens() or pseudoinverse(); pivots that are not positive are skipped
    Vector4<T> solveLDLT( const Vector4<T> & b ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );

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
    constexpr T eps2 = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon() / 64;
    // the pairs (p,q) of each rotation and the two other indices r
    constexpr int P[6] = { 0, 0, 0, 1, 1, 2 }, Q[6] = { 1, 2, 3, 2, 3, 3 }, R1[6] = { 2, 1, 1, 0, 0, 0 }, R2[6] = { 3, 3, 2, 3, 2, 1 };
    for ( int sweep = 0; sweep < 32; ++sweep )
    {
        bool rotated = false;
        for ( int i = 0; i < 6; ++i )
        {
            const int p = P[i], q = Q[i];
            const T apq = a[p][q];
            // the rotation is skipped if it cannot change the diagonal elements
            if ( apq * apq <= eps2 * std::abs( a[p][p] * a[q][q] ) )
            {
                a[p][q] = a[q][p] = 0;
                continue;
            }
            rotated = true;
            // t = tan, c = cos, s = sin of the rotation angle
            const T d = a[q][q] - a[p][p];
            T t, c;
            if ( std::abs( d ) > T( 1 << 28 ) * std::abs( apq ) )
            {
                // tiny angle: t = apq / d and c = 1 within the precision
                t = apq / d;
                c = 1;
            }
            else
            {
                // c is found from h and u, not from t, so that the two square roots do not wait for one another
                const T h = std::sqrt( d * d + 4 * apq * apq );
                const T u = std::abs( d ) + h;
                t = 2 * apq / std::copysign( u, d );
                c = std::sqrt( u / ( 2 * h ) );
            }
            const T s = t * c;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = a[q][p] = 0;
            for ( int k : { R1[i], R2[i] } )
            {
                const T akp = a[k][p], akq = a[k][q];
                a[k][p] = a[p][k] = c * akp - s * akq;
                a[k][q] = a[q][k] = s * akp + c * akq;
            }
            const Vector4<T> vp = v[p], vq = v[q];
            v[p] = c * vp - s * vq;
            v[q] = s * vp + c * vq;
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

template <typename T>
Vector4<T> SymMatrix4<T>::solveLDLT( const Vector4<T> & b ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // M = L*D*L^T with unit lower-triangular L and diagonal D = ( d0, d1, d2, d3 ); eij = lij * dj
    // a pivot that is not positive is skipped: the solution stays finite for a degenerate positive semi-definite M
    const auto inv = []( T d ) { return d > 0 ? 1 / d : T( 0 ); };
    const T i0 = inv( xx );
    const T l10 = xy * i0, l20 = xz * i0, l30 = xw * i0;
    const T i1 = inv( yy - l10 * xy );
    const T e21 = yz - l20 * xy, e31 = yw - l30 * xy;
    const T l21 = e21 * i1, l31 = e31 * i1;
    const T i2 = inv( zz - l20 * xz - l21 * e21 );
    const T e32 = zw - l30 * xz - l31 * e21;
    const T l32 = e32 * i2;
    const T i3 = inv( ww - l30 * xw - l31 * e31 - l32 * e32 );
    // L*y = b
    const T y0 = b.x;
    const T y1 = b.y - l10 * y0;
    const T y2 = b.z - l20 * y0 - l21 * y1;
    const T y3 = b.w - l30 * y0 - l31 * y1 - l32 * y2;
    // D*L^T*x = y
    Vector4<T> x;
    x.w = y3 * i3;
    x.z = y2 * i2 - l32 * x.w;
    x.y = y1 * i1 - l21 * x.z - l31 * x.w;
    x.x = y0 * i0 - l10 * x.y - l20 * x.z - l30 * x.w;
    return x;
}

/// \}

} // namespace MR
