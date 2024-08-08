#include "MRBestFitPolynomial.h"
#include "MRGTest.h"

#include <Eigen/Dense>

#include <cmath>
#include <complex>

namespace
{


template <typename T, size_t degree>
struct Solver
{
    Eigen::Vector<std::complex<T>, degree> operator() ( const Eigen::Vector<T, degree + 1>& coeffs ) = delete;
};

template <typename T>
struct Solver<T, 1>
{
    Eigen::Vector<std::complex<T>, 1> operator() ( const Eigen::Vector<T, 2>& c )
    {
        assert( c[1] != 0 );
        // y(x) = c[0] + c[1] * x = 0 => x = -c[0] / c[1]
        return Eigen::Vector<std::complex<T>, 1>{ -c[0] / c[1] };
    }
};

template <typename T>
struct Solver<T, 2>
{
    Eigen::Vector<std::complex<T>, 2> operator() ( const Eigen::Vector<T, 3>& coeffs )
    {
        assert( coeffs[2] != 0 );

        // y(x) = c[0] + c[1] * x + c[2] * x^2 = 0
        const auto b = coeffs[1] / coeffs[2];
        const auto c = coeffs[0] / coeffs[2];

        const auto D = std::sqrt( std::complex<T>{ b * b - T(4) * c } );
        return { ( -b + D ) / T( 2 ), ( -b - D ) / T( 2 ) };
    }
};

template <typename T>
struct Solver<T, 3>
{
    Eigen::Vector<std::complex<T>, 3> operator() ( const Eigen::Vector<T, 4>& coeffs )
    {
        assert( coeffs[3] != 0 );
        const auto& a = coeffs[3];
        const auto& b = coeffs[2];
        const auto& c = coeffs[1];
        const auto& d = coeffs[0];

        const T p = ( T(3)*a*c - b*b ) / ( T(3)*a*a );
        const T q
            = ( T(2)*b*b*b - T(9)*a*b*c + T(27)*a*a*d ) / ( T(27) * a*a*a );
        const T alpha = b / ( T(3) * a );

        const std::complex<T> D = q*q / T(4) + p*p*p / T(27);
        const auto Ds = std::sqrt( D );

        const std::complex<T> e1 = ( std::complex<T>( T(-1), sqrt( T(3) ) ) ) / T(2);
        const std::complex<T> e2 = ( std::complex<T>( T(-1), - sqrt( T(3) ) ) ) / T(2);

        const auto u1s = std::pow( -q / T(2) + Ds, T(1) / T(3) );
        const auto u2s = std::pow( -q / T(2) - Ds, T(1) / T(3) );

        return { u1s + u2s - alpha, e1*u1s + e2*u2s - alpha, e2*u1s + e1*u2s - alpha };
    }
};


template <typename T>
struct Solver<T, 4>
{
    Eigen::Vector<std::complex<T>, 4> operator()( const Eigen::Vector<T, 5>& coeffs )
    {
        const auto A = coeffs[4];
        const auto B = coeffs[3];
        const auto C = coeffs[2];
        const auto D = coeffs[1];
        const auto E = coeffs[0];

        const auto alpha = B / ( T(4) * A );
        // depressed equation
        const auto a = -T(3)*B*B / ( T(8)*A*A ) + C/A;
        const auto b = B*B*B / ( T(8)*A*A*A ) - B*C / ( T(2)*A*A ) + D/A;
        const auto c = -T(3)*B*B*B*B / ( T(256)*A*A*A*A ) + C*B*B / ( T(16)*A*A*A ) - B*D/(T(4)*A*A) + E/A;

        // bi-quadratic
        if ( std::abs( b ) < T( 0.0001 ) )
        {
            Solver<T, 2> solver2;
            const auto ys = solver2( { c, a, T(1) } );
            // u*u == y => u = std::sqrt(y)
            return {
                std::sqrt( ys[0] ) - alpha,
                -std::sqrt( ys[0] ) - alpha,
                std::sqrt( ys[1] ) - alpha,
                -std::sqrt( ys[1] ) - alpha
            };
        }
        else
        {
            Solver<T, 3> solver3;
            const auto ys = solver3( { a*c - T(0.25)*b*b, -T(2)*c, -a, T(2) } );
            const auto y = ys[0];

            const auto t2 = std::complex<T>{ T(2)*y - a };
            const auto mt2 = std::complex<T>{ -T(2)*y - a };
            const auto t = std::sqrt( t2 );

            return {
                T(0.5) * ( -t + std::sqrt( mt2 + T(2)*b/t ) ) - alpha,
                T(0.5) * ( -t - std::sqrt( mt2 + T(2)*b/t ) ) - alpha,
                T(0.5) * ( t + std::sqrt( mt2 - T(2)*b/t ) ) - alpha,
                T(0.5) * ( t - std::sqrt( mt2 - T(2)*b/t ) ) - alpha
            };
        }
    }
};


template <typename T, size_t degree>
std::vector<std::complex<T>> solveWithFixedDegree( auto coeffsBegin, auto coeffsEnd )
{
    assert( coeffsEnd - coeffsBegin == degree + 1 );
    if constexpr ( degree == 0 )
    {
        return {};
    }
    else
    {
        if ( *std::prev( coeffsEnd ) != 0 )
        {
            Eigen::Vector<T, degree + 1> c;
            for ( size_t i = 0; i < degree + 1; ++i )
                c[i] = *( coeffsBegin + i );

            Solver<T, degree> s;
            auto roots = s( c );

            std::vector<std::complex<T>> res;
            for ( auto v : roots )
                res.push_back( v );
            return res;
        }
        else
        {
            return solveWithFixedDegree<T, degree - 1>( coeffsBegin, std::prev( coeffsEnd ) );
        }
    }
}



}

namespace MR
{

template <typename T, size_t degree>
T Polynomial<T, degree>::operator()( T x ) const
{
    T res = 0;
    T xn = 1;
    for ( T v : a )
    {
        res += v * xn;
        xn *= x;
    }

    return res;
}

template <typename T, size_t degree>
std::vector<T> Polynomial<T, degree>::solve( T tol ) const
    requires ( canSolvePolynomial( degree ) )
{
    // workaround for old clang versions
#if defined( __clang__ ) && ( __clang_major__ < 15 )
    if constexpr ( canSolvePolynomial( degree ) )
    {
#endif
    auto r_c = solveWithFixedDegree<T, degree>( a.begin(), a.end() );
    std::vector<T> r;
    for ( std::complex<T> c : r_c )
        if ( std::abs( c.imag() ) < tol )
            r.push_back( c.real() );
    return r;
#if defined( __clang__ ) && ( __clang_major__ < 15 )
    }
    else
        return {};
#endif
}

template <typename T, size_t degree>
Polynomial<T, degree == 0 ? 0 : degree - 1> Polynomial<T, degree>::deriv() const
{
    if constexpr ( degree == 0 )
    {
        return Polynomial<T, 0>{ Eigen::Vector<T, 1>{ T{} } };
    }
    else
    {
        Eigen::Vector<T, degree> r;
        for ( size_t i = 1; i < n; ++i )
            r[i - 1] = i * a[i];
        return { r };
    }
}

template <typename T, size_t degree>
T Polynomial<T, degree>::intervalMin( T l, T r ) const
    requires ( canMinimizePolynomial( degree ) )
{
    auto eval = [this] ( T x )
    {
        return ( *this ) ( x );
    };
    auto argmin = [eval] ( T x1, T x2 )
    {
        return eval( x1 ) < eval( x2 ) ? x1 : x2;
    };

    T mn = argmin( l, r );
    T mnVal = eval( mn );

    // workaround for old clang versions
#if defined( __clang__ ) && ( __clang_major__ < 15 )
    if constexpr ( canMinimizePolynomial( degree ) )
    {
#endif
    if constexpr ( degree >= 2 )
    {
        const auto candidates = deriv().solve( T( 0.0001 ) );
        for ( auto root : candidates )
        {
            auto v = eval( root );
            if ( l <= root && root <= r && v < mnVal )
            {
                mn = root;
                mnVal = v;
            }
        }
    }
#if defined( __clang__ ) && ( __clang_major__ < 15 )
    }
#endif

    return mn;
}

template struct Polynomial<float, 0>;
template struct Polynomial<float, 1>;
template struct Polynomial<float, 2>;
template struct Polynomial<float, 3>;
template struct Polynomial<float, 4>;
template struct Polynomial<float, 5>;
template struct Polynomial<float, 6>;

template struct Polynomial<double, 0>;
template struct Polynomial<double, 1>;
template struct Polynomial<double, 2>;
template struct Polynomial<double, 3>;
template struct Polynomial<double, 4>;
template struct Polynomial<double, 5>;
template struct Polynomial<double, 6>;


template <typename T>
T PolynomialWrapper<T>::operator()( T x ) const
{
    return std::visit( [x] ( const auto& p )
    {
        return p( x );
    }, poly );
}

template <typename T>
PolynomialWrapper<T> PolynomialWrapper<T>::deriv() const
{
    return std::visit( [] ( const auto& p ) -> PolynomialWrapper<T>
    {
        return p.deriv();
    }, poly );
}

template <typename T>
std::optional<T> PolynomialWrapper<T>::intervalMin( T a, T b ) const
{
    return std::visit( overloaded{
        [a, b] <size_t degree> ( const Polynomial<T, degree>& p ) -> std::optional<T>
        {
            if constexpr ( canMinimizePolynomial( degree ) )
                return p.intervalMin( a, b );
            else
                return std::nullopt;
        }
    }, poly );
}

template struct PolynomialWrapper<float>;
template struct PolynomialWrapper<double>;


template <typename T, size_t degree>
BestFitPolynomial<T, degree>::BestFitPolynomial( T reg ):
    lambda_( reg ),
    XtX_( Eigen::Matrix<T, n, n>::Zero() ),
    XtY_( Eigen::Vector<T, n>::Zero() )
{}

template <typename T, size_t degree>
void BestFitPolynomial<T, degree>::addPoint( T x, T y )
{
    // n-th power of x
    Eigen::Vector<T, n> xs;
    T xn = 1;
    for ( size_t i = 0; i < n; ++i )
    {
        xs[i] = xn;
        xn *= x;
    }

    XtX_ += xs * xs.transpose();
    XtY_ += y * xs;
    sumWeight_ += T( 1 );
}

template <typename T, size_t degree>
void BestFitPolynomial<T, degree>::addPoint( T x, T y, T weight )
{
    // n-th power of x
    Eigen::Vector<T, n> xs;
    T xn = 1;
    for ( size_t i = 0; i < n; ++i )
    {
        xs[i] = xn;
        xn *= x;
    }

    XtX_ += weight * xs * xs.transpose();
    XtY_ += weight * y * xs;
    sumWeight_ += weight;
}


template <typename T, size_t degree>
Polynomial<T, degree> BestFitPolynomial<T, degree>::getBestPolynomial() const
{
    const Eigen::Matrix<T, n, n> m = XtX_ + sumWeight_ * lambda_ * Eigen::Matrix<T, n, n>::Identity();
    const Eigen::Vector<T, n> w = m.fullPivLu().solve( XtY_ );
    return { w };
}


template class BestFitPolynomial<float, 2>;
template class BestFitPolynomial<float, 3>;
template class BestFitPolynomial<float, 4>;
template class BestFitPolynomial<float, 5>;
template class BestFitPolynomial<float, 6>;

template class BestFitPolynomial<double, 2>;
template class BestFitPolynomial<double, 3>;
template class BestFitPolynomial<double, 4>;
template class BestFitPolynomial<double, 5>;
template class BestFitPolynomial<double, 6>;



TEST( MRMesh, BestFitPolynomial )
{
    const std::vector<double> xs {
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
    };
    const std::vector<double> ys {
        0.0810,
        0.0786689,
        0.00405961,
        -0.0595083,
        -0.0204948,
        0.00805113,
        -0.0037784,
        -0.00639334,
        -0.00543275,
        -0.00722818,
        0
    };

    // expected coefficients
    const std::vector<double> alpha
    {
        -0.000782968,
        0.0221325,
        -0.0110833,
        -0.00346209,
        0.00145959,
        8.99359e-05,
        -3.8049e-05,
    };

    assert( xs.size() == ys.size() );

    BestFitPolynomial<double, 6> bestFit( 0.0 );
    for ( size_t i = 0; i < xs.size(); ++i )
        bestFit.addPoint( xs[i], ys[i] );

    const auto poly = bestFit.getBestPolynomial();

    ASSERT_EQ( poly.a.size(), (int)alpha.size() );
    for ( size_t i = 0; i < alpha.size(); ++i )
        ASSERT_NEAR( poly.a[i], alpha[i], 0.000001 );
}

TEST( MRMesh, PolynomialRoots1 )
{
    Polynomialf<1> p{ { 3.f, 2.f } };
    auto roots = p.solve( 0.0001f );
    ASSERT_EQ( roots.size(), 1ull );
    ASSERT_NEAR( roots[0], -1.5f, 0.001f );
}

TEST( MRMesh, PolynomialRoots2 )
{
    Polynomialf<2> p{ { -1.f, 2.f, 1.f } };
    auto roots = p.solve( 0.0001f );
    ASSERT_EQ( roots.size(), 2ull );
    std::sort( roots.begin(), roots.end() );
    ASSERT_NEAR( roots[0], -2.414f, 0.001f );
    ASSERT_NEAR( roots[1], 0.414f, 0.001f );
}

TEST( MRMesh, PolynomialRoots3 )
{
    Polynomialf<3> p{ { -2.f, 0.2f, 3.f, 1.f } };
    auto roots = p.solve( 0.0001f );
    ASSERT_EQ( roots.size(), 3ull );
    std::sort( roots.begin(), roots.end() );
    ASSERT_NEAR( roots[0], -2.636f, 0.001f );
    ASSERT_NEAR( roots[1], -1.072f, 0.001f );
    ASSERT_NEAR( roots[2], 0.708f, 0.001f );
}

TEST( MRMesh, PolynomialRoots4 )
{
    Polynomialf<4> p{ { -2.f, 0.3f, 4.f, -0.1f, -1.f } };
    auto roots = p.solve( 0.0001f );
    ASSERT_EQ( roots.size(), 4ull );
    std::sort( roots.begin(), roots.end() );
    ASSERT_NEAR( roots[0], -1.856f, 0.001f );
    ASSERT_NEAR( roots[1], -0.809f, 0.001f );
    ASSERT_NEAR( roots[2], 0.724f, 0.001f );
    ASSERT_NEAR( roots[3], 1.841f, 0.001f );
}

TEST( MRMesh, PolynomialRoots4_biquadratic )
{
    Polynomialf<4> p{ { 23.f, -40.f, 26.f, -8.f, 1.f } };
    auto roots = p.solve( 0.0001f );
    ASSERT_EQ( roots.size(), 2ull );
    std::sort( roots.begin(), roots.end() );
    ASSERT_NEAR( roots[0], 1.356f, 0.001f );
    ASSERT_NEAR( roots[1], 2.644f, 0.001f );
}

TEST( MRMesh, PolynomialRoots )
{
    const std::vector<double> xs {
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
    };
    const std::vector<double> ys {
        0.0810,
        0.0786689,
        0.00405961,
        -0.0595083,
        -0.0204948,
        0.00805113,
        -0.0037784,
        -0.00639334,
        -0.00543275,
        -0.00722818,
        0
    };
    assert( xs.size() == ys.size() );

    BestFitPolynomial<double, 6> bestFit( 0.0 );
    for ( size_t i = 0; i < xs.size(); ++i )
        bestFit.addPoint( xs[i], ys[i] );

    const auto poly = bestFit.getBestPolynomial();
    const auto deriv = poly.deriv();
    const auto mn = deriv.intervalMin( -4.5, 4.5 );
    ASSERT_NEAR( mn, -3.629f, 0.001f );
}


}


