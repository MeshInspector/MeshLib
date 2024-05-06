#pragma once

#include "MRAffineXf3.h"
#include "MRQuaternion.h"

namespace MR
{

/// rigid transformation preserving all distances: y = A*x + b,
/// where A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
template <typename T>
struct RigidXf3
{
    using V = Vector3<T>;

    V a; ///< rotation angles relative to x,y,z axes
    V b; ///< shift

    constexpr RigidXf3() noexcept = default;
    constexpr RigidXf3( const V & a, const V & b ) noexcept : a( a ), b( b ) { }
    template <typename U>
    constexpr explicit RigidXf3( const RigidXf3<U> & xf ) noexcept : a( xf.a ), b( xf.b ) { }

    /// converts this into rigid transformation, which non-linearly depends on angles
    [[nodiscard]] AffineXf3<T> rigidXf() const { return { Matrix3<T>( Quaternion<T>( a, a.length() ) ), b }; }

    /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
    [[nodiscard]] AffineXf3<T> linearXf() const { return { Matrix3<T>::approximateLinearRotationMatrixFromEuler( a ), b }; }
};

} // namespace MR
