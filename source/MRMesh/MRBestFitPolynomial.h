#pragma once

#include "MRMeshFwd.h"
#include <MRPch/MREigenCore.h>

#include <cstddef>
#include <vector>


namespace MR
{

// Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
// The following degrees are instantiated: [2; 6].

template <typename T, size_t degree>
struct Polynomial
{
    static constexpr bool canSolve = degree <= 4;
    static constexpr bool canSolveDerivative = degree <= 5;

    static constexpr size_t derivDegree = degree == 0 ? 0 : degree - 1;
    static constexpr size_t n = degree + 1;
    Eigen::Vector<T, n> a;

    template <typename NewT>
    Polynomial<NewT, degree> cast()
    { return { a.template cast<NewT>() }; }

    MRMESH_API T operator() ( T x ) const;

    MRMESH_API std::vector<T> solve( T tol ) const
        requires canSolve;

    MRMESH_API Polynomial<T, derivDegree> deriv() const;

    MRMESH_API T intervalMin( T a, T b ) const
        requires canSolveDerivative;
};

template <size_t degree>
using Polynomialf = Polynomial<float, degree>;

template <size_t degree>
using Polynomiald = Polynomial<double, degree>;


template <typename T, size_t degree>
class BestFitPolynomial
{
public:
    /// @param reg Regularization term (L2-reg)
    MRMESH_API explicit BestFitPolynomial( T reg );

    MRMESH_API void addPoint( T x, T y );

    MRMESH_API void addPoint( T x, T y, T weight );

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