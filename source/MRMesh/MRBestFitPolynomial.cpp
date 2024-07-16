#include "MRBestFitPolynomial.h"
#include "MRGTest.h"

#include <MRPch/MRSpdlog.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>

#include <spdlog/fmt/ostr.h>

#include <cmath>
#include <complex>

namespace
{


const double PI = 3.141592653589793238463L;
const double M_2PI = 2*PI;
const double eps=1e-12;

typedef std::complex<double> DComplex;

//---------------------------------------------------------------------------
// useful for testing
inline DComplex polinom_2(DComplex x, double a, double b)
{
    //Horner's scheme for x*x + a*x + b
    return x * (x + a) + b;
}

//---------------------------------------------------------------------------
// useful for testing
inline DComplex polinom_3(DComplex x, double a, double b, double c)
{
    //Horner's scheme for x*x*x + a*x*x + b*x + c;
    return x * (x * (x + a) + b) + c;
}

//---------------------------------------------------------------------------
// useful for testing
inline DComplex polinom_4(DComplex x, double a, double b, double c, double d)
{
    //Horner's scheme for x*x*x*x + a*x*x*x + b*x*x + c*x + d;
    return x * (x * (x * (x + a) + b) + c) + d;
}


//---------------------------------------------------------------------------
// solve cubic equation x^3 + a*x^2 + b*x + c
// x - array of size 3
// In case 3 real roots: => x[0], x[1], x[2], return 3
//         2 real roots: x[0], x[1],          return 2
//         1 real root : x[0], x[1] ± i*x[2], return 1
unsigned int solveP3(double *x,double a,double b,double c) {
    double a2 = a*a;
    double q  = (a2 - 3*b)/9;
    double r  = (a*(2*a2-9*b) + 27*c)/54;
    double r2 = r*r;
    double q3 = q*q*q;
    double A,B;
    if(r2<q3)
    {
        double t=r/sqrt(q3);
        if( t<-1) t=-1;
        if( t> 1) t= 1;
        t=acos(t);
        a/=3; q=-2*sqrt(q);
        x[0]=q*cos(t/3)-a;
        x[1]=q*cos((t+M_2PI)/3)-a;
        x[2]=q*cos((t-M_2PI)/3)-a;
        return 3;
    }
    else
    {
        A =-pow(fabs(r)+sqrt(r2-q3),1./3);
        if( r<0 ) A=-A;
        B = (0==A ? 0 : q/A);

        a/=3;
        x[0] =(A+B)-a;
        x[1] =-0.5*(A+B)-a;
        x[2] = 0.5*sqrt(3.)*(A-B);
        if(fabs(x[2])<eps) { x[2]=x[1]; return 2; }

        return 1;
    }
}

//---------------------------------------------------------------------------
// Solve quartic equation x^4 + a*x^3 + b*x^2 + c*x + d
// (attention - this function returns dynamically allocated array. It has to be released afterwards)
DComplex* solve_quartic(double a, double b, double c, double d)
{
    double a3 = -b;
    double b3 =  a*c -4.*d;
    double c3 = -a*a*d - c*c + 4.*b*d;

    // cubic resolvent
    // y^3 − b*y^2 + (ac−4d)*y − a^2*d−c^2+4*b*d = 0

    double x3[3];
    unsigned int iZeroes = solveP3(x3, a3, b3, c3);

    double q1, q2, p1, p2, D, sqD, y;

    y = x3[0];
    // THE ESSENCE - choosing Y with maximal absolute value !
    if(iZeroes != 1)
    {
        if(fabs(x3[1]) > fabs(y)) y = x3[1];
        if(fabs(x3[2]) > fabs(y)) y = x3[2];
    }

    // h1+h2 = y && h1*h2 = d  <=>  h^2 -y*h + d = 0    (h === q)

    D = y*y - 4*d;
    if(fabs(D) < eps) //in other words - D==0
    {
        q1 = q2 = y * 0.5;
        // g1+g2 = a && g1+g2 = b-y   <=>   g^2 - a*g + b-y = 0    (p === g)
        D = a*a - 4*(b-y);
        if(fabs(D) < eps) //in other words - D==0
            p1 = p2 = a * 0.5;

        else
        {
            sqD = sqrt(D);
            p1 = (a + sqD) * 0.5;
            p2 = (a - sqD) * 0.5;
        }
    }
    else
    {
        sqD = sqrt(D);
        q1 = (y + sqD) * 0.5;
        q2 = (y - sqD) * 0.5;
        // g1+g2 = a && g1*h2 + g2*h1 = c       ( && g === p )  Krammer
        p1 = (a*q1-c)/(q1-q2);
        p2 = (c-a*q2)/(q1-q2);
    }

    DComplex* retval = new DComplex[4];

    // solving quadratic eq. - x^2 + p1*x + q1 = 0
    D = p1*p1 - 4*q1;
    if(D < 0.0)
    {
        retval[0].real( -p1 * 0.5 );
        retval[0].imag( sqrt(-D) * 0.5 );
        retval[1] = std::conj(retval[0]);
    }
    else
    {
        sqD = sqrt(D);
        retval[0].real( (-p1 + sqD) * 0.5 );
        retval[1].real( (-p1 - sqD) * 0.5 );
    }

    // solving quadratic eq. - x^2 + p2*x + q2 = 0
    D = p2*p2 - 4*q2;
    if(D < 0.0)
    {
        retval[2].real( -p2 * 0.5 );
        retval[2].imag( sqrt(-D) * 0.5 );
        retval[3] = std::conj(retval[2]);
    }
    else
    {
        sqD = sqrt(D);
        retval[2].real( (-p2 + sqD) * 0.5 );
        retval[3].real( (-p2 - sqD) * 0.5 );
    }

    return retval;
}



}


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
        assert( c[2] != 0 );

        // y(x) = c[0] + c[1] * x + c[2] * x^2 = 0
        const auto b = coeffs[1] / coeffs[2];
        const auto c = coeffs[0] / coeffs[1];

        const auto D = std::sqrt( std::complex<T>{ b * b - 4 * c } );
        return { ( -b + D ) / T( 2 ), ( -b - D ) / T( 2 ) };
    }
};

template <typename T>
struct Solver<T, 3>
{
    Eigen::Vector<std::complex<T>, 3> operator() ( const Eigen::Vector<T, 4>& coeffs )
    {
        assert( c[3] != 0 );
        const T p = ( 3 * coeffs[3] * coeffs[1] - coeffs[2]*coeffs[2] ) / ( 3 * coeffs[3] * coeffs[3] );
        const T q
            = ( 2 * coeffs[2] * coeffs[2] - 9 * coeffs[3] * coeffs[2] * coeffs[1] + 27 * coeffs[3] * coeffs[3] * coeffs[0] )
                / ( 27 * coeffs[3] * coeffs[3] * coeffs[3] );
        const T alpha = -coeffs[2] / ( 3 * coeffs[3] );

        const std::complex<T> D = q*q / T(4) + p*p*p / T(27);
        const auto Ds = std::sqrt( D );

        const std::complex<T> e1 = ( std::complex<T>( T(-1), sqrt( T(3) ) ) ) / T(2);
        const std::complex<T> e2 = ( std::complex<T>( T(-1), - sqrt( T(3) ) ) ) / T(2);

        const auto u1s = std::pow( -q / T(2) + Ds, 1 / T(3) );
        const auto u2s = std::pow( -q / T(2) - Ds, 1 / T(3) );

        return { u1s + u2s + alpha, e1*u1s + e2*u2s + alpha, e2*u1s + e1*u2s + alpha };
    }
};


template <typename T>
struct Solver<T, 4>
{
    Eigen::Vector<std::complex<T>, 4> operator()( const Eigen::Vector<T, 5>& coeffs )
    {
        double a = coeffs[1] / coeffs[0];
        double b = coeffs[2] / coeffs[0];
        double c = coeffs[3] / coeffs[0];
        double d = coeffs[4] / coeffs[0];

        const auto s = solve_quartic( a, b, c, d );
        return
             Eigen::Vector<std::complex<double>, 4>{ s[0], s[1], s[2], s[3] }
            .cast<std::complex<T>>();
    }
};


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
    requires canSolve
{
//    Solver<T, degree> solver;
//    auto r_c = solver( a );
//    std::vector<T> r;
//    for ( std::complex<T> c : r_c )
//        if ( c.imag() < tol )
//            r.push_back( c.real() );
//    return r;

    // TODO: implement solvers for every possible degree
    Eigen::PolynomialSolver<T, degree> solver;
    solver.compute( a );
    std::vector<T> r;
    solver.realRoots( r, tol );
    return r;
}

template <typename T, size_t degree>
Polynomial<T, degree - 1> Polynomial<T, degree>::deriv() const
    requires ( degree >= 1 )
{
    Eigen::Vector<T, degree> r;
    for ( size_t i = 1; i < n; ++i )
        r[i - 1] = i * a[i];
    return { r };
}

template <typename T, size_t degree>
std::vector<T> Polynomial<T, degree>::localMins() const
    requires canSolveDerivative
{
    return {};
}

template <typename T, size_t degree>
std::vector<T> Polynomial<T, degree>::localMaxs() const
    requires canSolveDerivative
{
    return {};
}

template <typename T, size_t degree>
T Polynomial<T, degree>::intervalMin( T a, T b ) const
    requires canSolveDerivative
{
    auto eval = [this] ( T x )
    {
        return ( *this ) ( x );
    };
    auto argmin = [this, eval] ( T x1, T x2 )
    {
        return eval( x1 ) < eval( x2 ) ? x1 : x2;
    };

    T mn = argmin( a, b );
    T mnVal = eval( mn );

    const auto candidates = deriv().solve( T( 0.0001 ) );
    for ( auto r : candidates )
    {
        auto v = eval( r );
        spdlog::info( "Candidate: x={}, p(x)={}", r, v );
        if ( v < mnVal )
        {
            mn = r;
            mnVal = v;
        }
    }

    return mn;
}

template <typename T, size_t degree>
T Polynomial<T, degree>::intervalMax( T, T ) const
    requires canSolveDerivative
{
    return {};
}

template struct Polynomial<float, 2>;
template struct Polynomial<float, 3>;
template struct Polynomial<float, 4>;
template struct Polynomial<float, 5>;
template struct Polynomial<float, 6>;
//
//template struct Polynomial<double, 2>;
//template struct Polynomial<double, 3>;
//template struct Polynomial<double, 4>;
//template struct Polynomial<double, 5>;
template struct Polynomial<double, 6>;


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
    ++N_;
}


template <typename T, size_t degree>
Polynomial<T, degree> BestFitPolynomial<T, degree>::getBestPolynomial() const
{
    const Eigen::Matrix<T, n, n> m = XtX_ + static_cast<float>( N_ ) * lambda_ * Eigen::Matrix<T, n, n>::Identity();
    const Eigen::Vector<T, n> w = m.fullPivLu().solve( XtY_ );
    return { w };
}


//template class BestFitPolynomial<float, 2>;
//template class BestFitPolynomial<float, 3>;
//template class BestFitPolynomial<float, 4>;
//template class BestFitPolynomial<float, 5>;
template class BestFitPolynomial<float, 6>;
//
//template class BestFitPolynomial<double, 2>;
//template class BestFitPolynomial<double, 3>;
//template class BestFitPolynomial<double, 4>;
//template class BestFitPolynomial<double, 5>;
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

    ASSERT_EQ( poly.a.size(), alpha.size() );
    for ( size_t i = 0; i < alpha.size(); ++i )
        ASSERT_NEAR( poly.a[i], alpha[i], 0.000001 );
}


}


