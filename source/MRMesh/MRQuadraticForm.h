#pragma once

#include "MRSymMatrix2.h"
#include "MRSymMatrix3.h"

namespace MR
{

/// quadratic form: f = x^T A x + c
/// \ingroup MathGroup
template <typename V>
struct QuadraticForm
{
    using T = typename V::ValueType;
    using SM = typename V::SymMatrixType;

    SM A;
    T c = 0;
    
    /// evaluates the function at given x
    T eval( const V & x ) const { return dot( x, A * x ) + c; }

    /// adds to this squared distance to the origin point
    void addDistToOrigin( T weight ) { A += SM::diagonal( weight ); }

    /// adds to this squared distance to plane passing via origin with given unit normal
    void addDistToPlane( const V & planeUnitNormal )           { A +=          outerSquare( planeUnitNormal ); }
    void addDistToPlane( const V & planeUnitNormal, T weight ) { A += weight * outerSquare( planeUnitNormal ); }

    /// adds to this squared distance to line passing via origin with given unit direction
    void addDistToLine( const V & lineUnitDir )           { A +=            SM::identity() - outerSquare( lineUnitDir ); }
    void addDistToLine( const V & lineUnitDir, T weight ) { A += weight * ( SM::identity() - outerSquare( lineUnitDir ) ); }
};

/// given two quadratic forms with points where they reach minima,
/// computes sum quadratic form and the point where it reaches minimum
/// \related QuadraticForm
template <typename V>
MRMESH_API std::pair< QuadraticForm<V>, V > sum(
    const QuadraticForm<V> & q0, const V & x0,
    const QuadraticForm<V> & q1, const V & x1, 
    bool minAmong01 = false ); ///< if true then the minimum is selected only between points x0 and x1

} // namespace MR
