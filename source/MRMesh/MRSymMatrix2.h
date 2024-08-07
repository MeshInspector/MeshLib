#pragma once

#include "MRVector2.h"
#include "MRMatrix2.h"
#include <limits>

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

    /// computes trace of the matrix
    constexpr T trace() const noexcept { return xx + yy; }
    /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
    constexpr T normSq() const noexcept { return sqr( xx ) + 2 * sqr( xy ) + sqr( yy ); }
    /// computes determinant of the matrix
    constexpr T det() const noexcept;
    /// computes inverse matrix
    constexpr SymMatrix2<T> inverse() const noexcept { return inverse( det() ); }
    /// computes inverse matrix given determinant of this
    constexpr SymMatrix2<T> inverse( T det ) const noexcept;

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

    /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
    /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
    /// M*V^T = V^T*L; M = V^T*L*V
    Vector2<T> eigens( Matrix2<T> * eigenvectors = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
    Vector2<T> eigenvector( T eigenvalue ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
    /// computes not-unit eigenvector corresponding to maximum eigenvalue
    Vector2<T> maxEigenvector() const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );

    /// for not-degenerate matrix returns just inverse matrix, otherwise
    /// returns degenerate matrix, which performs inversion on not-kernel subspace;
    /// \param tol relative epsilon-tolerance for too small number detection
    /// \param rank optional output for this matrix rank according to given tolerance
    /// \param space rank=1: unit direction of solution line, rank=2: zero vector
    SymMatrix2<T> pseudoinverse( T tol = std::numeric_limits<T>::epsilon(), int * rank = nullptr, Vector2<T> * space = nullptr ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> );
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

/// x = k * a * a^T
template <typename T>
inline SymMatrix2<T> outerSquare( T k, const Vector2<T> & a )
{
    const auto ka = k * a;
    SymMatrix2<T> res;
    res.xx = ka.x * a.x;
    res.xy = ka.x * a.y;
    res.yy = ka.y * a.y;
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
constexpr SymMatrix2<T> SymMatrix2<T>::inverse( T det ) const noexcept
{
    if ( det == 0 )
        return {};
    SymMatrix2<T> res;
    res.xx =  yy / det;
    res.xy = -xy / det;
    res.yy =  xx / det;
    return res;
}

template <typename T>
Vector2<T> SymMatrix2<T>::eigens( Matrix2<T> * eigenvectors ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    //https://en.wikipedia.org/wiki/Eigenvalue_algorithm#2%C3%972_matrices
    const auto tr = trace();
    const auto q = tr / 2;
    const auto p = std::sqrt( std::max( T(0), sqr( tr ) - 4 * det() ) ) / 2;
    Vector2<T> eig;
    if ( p <= std::abs( q ) * std::numeric_limits<T>::epsilon() )
    {
        // this is proportional to identity matrix
        eig = { q, q };
        if ( eigenvectors )
            *eigenvectors = Matrix2<T>{};
        return eig;
    }
    eig[0] = q - p;
    eig[1] = q + p;
    if ( eigenvectors )
    {
        const auto x = eigenvector( eig[0] ).normalized();
        *eigenvectors = Matrix2<T>::fromRows( x, x.perpendicular() );
    }
    return eig;
}

template <typename T>
Vector2<T> SymMatrix2<T>::eigenvector( T eigenvalue ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    const Vector2<T> row0( xx - eigenvalue, xy );
    const Vector2<T> row1( xy, yy - eigenvalue );
    // not-repeating eigenvalue means that one of two rows is not zero
    const T lsq0 = row0.lengthSq();
    const T lsq1 = row1.lengthSq();
    return lsq0 >= lsq1 ? row0.perpendicular() : row1.perpendicular();
}

template <typename T>
Vector2<T> SymMatrix2<T>::maxEigenvector() const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    const auto tr = trace();
    const auto q = tr / 2;
    const auto p = std::sqrt( std::max( T(0), sqr( tr ) - 4 * det() ) ) / 2;
    if ( p <= std::abs( q ) * std::numeric_limits<T>::epsilon() )
    {
        // this is proportional to identity matrix
        return Vector2<T>( T(1), T(0) );
    }
    return eigenvector( q + p );
}

template <typename T>
SymMatrix2<T> SymMatrix2<T>::pseudoinverse( T tol, int * rank, Vector2<T> * space ) const MR_REQUIRES_IF_SUPPORTED( std::is_floating_point_v<T> )
{
    SymMatrix2<T> res;
    Matrix2<T> eigenvectors;
    const auto eigenvalues = eigens( &eigenvectors );
    const auto threshold = std::max( std::abs( eigenvalues[0] ), std::abs( eigenvalues[1] ) ) * tol;
    int myRank = 0;
    for ( int i = 0; i < 2; ++i )
    {
        if ( std::abs( eigenvalues[i] ) <= threshold )
            continue;
        res += outerSquare( 1 / eigenvalues[i], eigenvectors[i] );
        ++myRank;
        if ( space )
        {
            if ( myRank == 1 )
                *space = eigenvectors[i];
            else
                *space = Vector2<T>{};
        }
    }
    if ( rank )
        *rank = myRank;
    return res;
}

/// \}

} // namespace MR
