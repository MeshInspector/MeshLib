#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <MRPch/MREigenCore.h>

#include <cstddef>
#include <vector>
#include <variant>
#include <optional>


namespace MR
{


constexpr bool canSolvePolynomial( auto degree )
{
    return 1 <= degree && degree <= 4;
}

constexpr bool canMinimizePolynomial( auto degree )
{
    return degree <= 5;
}

// Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
// The following degrees are instantiated: [2; 6].

template <typename T, size_t degree>
struct Polynomial
{
    static constexpr size_t n = degree + 1;

    // We're not binding Eigen at the moment, so this has to be hidden.
    MR_BIND_IGNORE Eigen::Vector<T, n> a;

    template <typename NewT>
    Polynomial<NewT, degree> cast() const
    { return { a.template cast<NewT>() }; }

    MRMESH_API T operator() ( T x ) const;

    MRMESH_API std::vector<T> solve( T tol ) const
        requires ( canSolvePolynomial( degree ) );

    MRMESH_API Polynomial<T, degree == 0 ? 0 : degree - 1> deriv() const;

    MRMESH_API T intervalMin( T a, T b ) const
        requires ( canMinimizePolynomial( degree ) );
};

template <size_t degree>
using Polynomialf = Polynomial<float, degree>;

template <size_t degree>
using Polynomiald = Polynomial<double, degree>;


template <typename T>
using Polynomialx = std::variant
    < Polynomial<T, 0>
    , Polynomial<T, 1>
    , Polynomial<T, 2>
    , Polynomial<T, 3>
    , Polynomial<T, 4>
    , Polynomial<T, 5>
    , Polynomial<T, 6>
    >;

/// This is a unifying interface for a polynomial of some degree, known only in runtime
template <typename T>
struct PolynomialWrapper
{
    Polynomialx<T> poly;

    template <size_t degree>
    PolynomialWrapper( const Polynomial<T, degree>& p ):
        poly( p )
    {}

    MRMESH_API T operator() ( T x ) const;

    MRMESH_API PolynomialWrapper<T> deriv() const;

    MRMESH_API std::optional<T> intervalMin( T a, T b ) const;
};

using PolynomialWrapperf = PolynomialWrapper<float>;
using PolynomialWrapperd = PolynomialWrapper<double>;


template <typename T, size_t degree>
class BestFitPolynomial
{
public:
    /// @param reg Regularization term (L2-reg)
    MRMESH_API explicit BestFitPolynomial( T reg );

    MRMESH_API void addPoint( T x, T y );

    MRMESH_API void addPoint( T x, T y, T weight );

    /// @note The result might have leading coefficient equal to zero.
    MRMESH_API Polynomial<T, degree> getBestPolynomial() const;

private:
    static constexpr size_t n = degree + 1;
    T lambda_ {};
    Eigen::Matrix<T, n, n> XtX_;
    Eigen::Vector<T, n> XtY_;
    T sumWeight_ = 0;
};

template <size_t degree>
using BestFitPolynomialf = BestFitPolynomial<float, degree>;

template <size_t degree>
using BestFitPolynomiald = BestFitPolynomial<double, degree>;



}
