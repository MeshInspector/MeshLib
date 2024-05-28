#pragma once

#include "MRRigidXf3.h"

namespace MR
{

/// rigid (with scale) transformation that multiplies all distances on same scale: y = s*A*x + b,
/// where s is a scalar, A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
template <typename T>
struct RigidScaleXf3
{
    using V = Vector3<T>;

    V a; ///< rotation angles relative to x,y,z axes
    V b; ///< shift
    T s = 1; ///< scaling

    constexpr RigidScaleXf3() noexcept = default;
    constexpr RigidScaleXf3( const V & a, const V & b, T s ) noexcept : a( a ), b( b ), s( s ) { }
    template <typename U>
    constexpr explicit RigidScaleXf3( const RigidScaleXf3<U> & xf ) noexcept : a( xf.a ), b( xf.b ), s( T( xf.s ) ) { }
    template <typename U>
    constexpr explicit RigidScaleXf3( const RigidXf3<U> & xf ) noexcept : a( xf.a ), b( xf.b ), s( 1 ) { }

    /// converts this into rigid (with scale) transformation, which non-linearly depends on angles
    [[nodiscard]] AffineXf3<T> rigidScaleXf() const { return { s * Matrix3<T>( Quaternion<T>( a, a.length() ) ), b }; }

    /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
    [[nodiscard]] AffineXf3<T> linearXf() const { return { s * Matrix3<T>::approximateLinearRotationMatrixFromEuler( a ), b }; }
};

} // namespace MR
