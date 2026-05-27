#include <MRMesh/MRBestFitPolynomial.h>
#include <MRMesh/MRGTest.h>

#include <algorithm>
#include <vector>

namespace MR
{

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

} //namespace MR
