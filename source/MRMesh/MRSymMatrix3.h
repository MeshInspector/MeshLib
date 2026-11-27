#pragma once

#include "MRVector3.h"
#include "MRMatrix3.h"
#include "MRConstants.h"
#include <cmath>
#include <limits>
#include <utility>

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

    // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
    //   when generating the bindings, and results in duplicate functions in C#.
    template <typename U> MR_REQUIRES_IF_SUPPORTED( !std::is_same_v<T, U> )
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
    constexpr SymMatrix3<T> inverse() const noexcept { return inverse( det() ); }
    /// computes inverse matrix given determinant of this
    constexpr SymMatrix3<T> inverse( T det ) const noexcept;

    SymMatrix3 & operator +=( const SymMatrix3<T> & b ) { xx += b.xx; xy += b.xy; xz += b.xz; yy += b.yy; yz += b.yz; zz += b.zz; return * this; }
    SymMatrix3 & operator -=( const SymMatrix3<T> & b ) { xx -= b.xx; xy -= b.xy; xz -= b.xz; yy -= b.yy; yz -= b.yz; zz -= b.zz; return * this; }
    constexpr SymMatrix3 & operator *=( T b ) { xx *= b; xy *= b; xz *= b; yy *= b; yz *= b; zz *= b; return * this; }
    constexpr SymMatrix3 & operator /=( T b )
    {
        if constexpr ( std::is_integral_v<T> )
            { xx /= b; xy /= b; xz /= b; yy /= b; yz /= b; zz /= b; return * this; }
        else
            return *this *= ( 1 / b );
    }

    /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
    /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
    /// M*V^T = V^T*L; M = V^T*L*V
    Vector3<T> eigens( Matrix3<T> * eigenvectors = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// same as eigens(), but by cyclic Jacobi method: somewhat slower, and precise also for tiny eigenvalues of an ill-conditioned matrix,
    /// where eigens() has the absolute error proportional to the largest eigenvalue
    Vector3<T> eigensJacobi( Matrix3<T> * eigenvectors = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
    Vector3<T> eigenvector( T eigenvalue ) const MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> );

    /// for not-degenerate matrix returns just inverse matrix, otherwise
    /// returns degenerate matrix, which performs inversion on not-kernel subspace;
    /// \param tol relative epsilon-tolerance for too small number detection
    /// \param rank optional output for this matrix rank according to given tolerance
    /// \param space rank=1: unit direction of solution line, rank=2: unit normal to solution plane, rank=3: zero vector
    SymMatrix3<T> pseudoinverse( T tol = std::numeric_limits<T>::epsilon(), int * rank = nullptr, Vector3<T> * space = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
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

/// x = k * a * a^T
template <typename T>
inline SymMatrix3<T> outerSquare( T k, const Vector3<T> & a )
{
    const auto ka = k * a;
    SymMatrix3<T> res;
    res.xx = ka.x * a.x;
    res.xy = ka.x * a.y;
    res.xz = ka.x * a.z;
    res.yy = ka.y * a.y;
    res.yz = ka.y * a.z;
    res.zz = ka.z * a.z;
    return res;
}

/// computes a matrix that gives double application of a cross product with given vector
/// M x = [ a, [a, x] ]
/// https://en.wikipedia.org/wiki/Cross_product#Alternative_ways_to_compute
template <typename T>
inline SymMatrix3<T> crossSquare( const Vector3<T> & a )
{
    SymMatrix3<T> res = outerSquare( a );
    const auto d = a.lengthSq();
    res.xx -= d;
    res.yy -= d;
    res.zz -= d;
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
constexpr SymMatrix3<T> SymMatrix3<T>::inverse( T det ) const noexcept
{
    if ( det == 0 )
        return {};
    SymMatrix3<T> res;
    res.xx = yy * zz - yz * yz;
    res.xy = xz * yz - xy * zz;
    res.xz = xy * yz - xz * yy;
    res.yy = xx * zz - xz * xz;
    res.yz = xz * xy - xx * yz;
    res.zz = xx * yy - xy * xy;
    res /= det; // one division for floating-point T
    return res;
}

template <typename T>
Vector3<T> SymMatrix3<T>::eigens( Matrix3<T> * eigenvectors ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
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
Vector3<T> SymMatrix3<T>::eigensJacobi( Matrix3<T> * eigenvectors ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // each rotation zeroes one off-diagonal element, see SymMatrix4::eigens
    T a[3][3] =
    {
        { xx, xy, xz },
        { xy, yy, yz },
        { xz, yz, zz }
    };
    Matrix3<T> v; // eigenvectors in rows
    constexpr T eps2 = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon() / 64;
    // the pair (p,q) of each rotation and the other index r
    constexpr int P[3] = { 0, 0, 1 }, Q[3] = { 1, 2, 2 }, R[3] = { 2, 1, 0 };
    for ( int sweep = 0; sweep < 32; ++sweep )
    {
        bool rotated = false;
        for ( int i = 0; i < 3; ++i )
        {
            const int p = P[i], q = Q[i], k = R[i];
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
                const T h = std::sqrt( d * d + 4 * apq * apq );
                const T u = std::abs( d ) + h;
                t = 2 * apq / std::copysign( u, d );
                c = std::sqrt( u / ( 2 * h ) );
            }
            const T s = t * c;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = a[q][p] = 0;
            const T akp = a[k][p], akq = a[k][q];
            a[k][p] = a[p][k] = c * akp - s * akq;
            a[k][q] = a[q][k] = s * akp + c * akq;
            const Vector3<T> vp = v[p], vq = v[q];
            v[p] = c * vp - s * vq;
            v[q] = s * vp + c * vq;
        }
        if ( !rotated )
            break;
    }
    int i0 = 0, i1 = 1, i2 = 2;
    if ( a[i0][i0] > a[i1][i1] )
        std::swap( i0, i1 );
    if ( a[i1][i1] > a[i2][i2] )
        std::swap( i1, i2 );
    if ( a[i0][i0] > a[i1][i1] )
        std::swap( i0, i1 );
    if ( eigenvectors )
        *eigenvectors = Matrix3<T>( v[i0], v[i1], v[i2] );
    return { a[i0][i0], a[i1][i1], a[i2][i2] };
}

template <typename T>
Vector3<T> SymMatrix3<T>::eigenvector( T eigenvalue ) const MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
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

namespace detail
{

/// cube root of the machine epsilon of T (std::cbrt is not constexpr before C++26);
/// for other types than float and double: eps = 2^(1 - digits), and the cube root is rounded up to a power of two
template <typename T>
inline constexpr T cCbrtEpsilon = T( 1 ) / T( 1ull << ( ( std::numeric_limits<T>::digits - 1 ) / 3 ) );
template <>
inline constexpr float cCbrtEpsilon<float> = 4.92156660e-3f;
template <>
inline constexpr double cCbrtEpsilon<double> = 6.0554544523933395e-6;

} //namespace detail

template <typename T>
SymMatrix3<T> SymMatrix3<T>::pseudoinverse( T tol, int * rank, Vector3<T> * space ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    // a positive definite matrix (all leading principal minors are positive) is inverted directly, which is several times faster,
    // if lambdaMin > tol * lambdaMax is guaranteed: lambdaMin >= det / m2 (m2 is the sum of principal 2x2 minors >= lambdaMax * lambdaMid),
    // and lambdaMax <= trace; rounding errors (about eps * trace^3) can make the determinant and the minors of a rank-1 matrix positive
    // and pass that check, so also det > tol^2 * (trace/3)^3 is required: it holds if lambdaMin > tol * lambdaMax, since then det > tol^2 * lambdaMax^3;
    // the tolerance here is at least cbrt(eps): the error of the direct inverse is about eps / tol <= eps^(2/3),
    // and tol^2 / 27 >= eps^(2/3) / 27 is far above the rounding errors of a rank-1 matrix
    if ( xx > 0 )
    {
        const auto mxy = xx * yy - xy * xy;
        if ( mxy > 0 )
        {
            // cofactors of the first row give the determinant, and together with the principal minors - the inverse
            const auto myz = yy * zz - yz * yz;
            const auto cxy = xz * yz - xy * zz;
            const auto cxz = xy * yz - xz * yy;
            const auto d = xx * myz + xy * cxy + xz * cxz;
            const auto mxz = xx * zz - xz * xz;
            const auto tr = trace();
            const auto dirTol = std::max( tol, detail::cCbrtEpsilon<T> );
            if ( d > 0 && d > dirTol * tr * ( mxy + mxz + myz ) && 27 * d > sqr( dirTol ) * tr * tr * tr )
            {
                if ( rank )
                    *rank = 3;
                if ( space )
                    *space = Vector3<T>{};
                // not res /= d after filling res: MSVC then reloads pairs of just stored doubles as vectors, which stalls store-to-load forwarding,
                // https://developercommunity.visualstudio.com/t/x64-O2:-scaling-a-just-filled-struct-of/11188541
                const auto k = 1 / d;
                SymMatrix3<T> res;
                res.xx = myz * k;
                res.xy = cxy * k;
                res.xz = cxz * k;
                res.yy = mxz * k;
                res.yz = ( xz * xy - xx * yz ) * k;
                res.zz = mxy * k;
                return res;
            }
        }
    }

    // Jacobi method keeps the small eigenvalues precise: eigens() can make them wrong in all digits or even not zero for a singular matrix
    SymMatrix3<T> res;
    Matrix3<T> eigenvectors;
    const auto eigenvalues = eigensJacobi( &eigenvectors );
    const auto threshold = std::max( std::abs( eigenvalues[0] ), std::abs( eigenvalues[2] ) ) * tol;
    int myRank = 0;
    for ( int i = 0; i < 3; ++i )
    {
        if ( std::abs( eigenvalues[i] ) <= threshold )
            continue;
        res += outerSquare( 1 / eigenvalues[i], eigenvectors[i] );
        ++myRank;
        if ( space )
        {
            if ( myRank == 1 )
                *space = eigenvectors[i];
            else if ( myRank == 2 )
                *space = cross( *space, eigenvectors[i] );
            else
                *space = Vector3<T>{};
        }
    }
    if ( rank )
        *rank = myRank;
    return res;
}

/// \}

} // namespace MR
