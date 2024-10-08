#pragma once

#include "MRVector2.h"
#include "MRVector3.h"
#include "MRMatrix2.h"
#include "MRMatrix3.h"
#include "MRAffineXf.h"

namespace MR
{

/// \defgroup MathGroup Math group

/// Conversion functions from lower-dimensional data to higher-dimensional put 0 in additional dimensions of vectors or identity elements in matrix
/// Conversion functions from higher-dimensional data to lower-dimensional just omit additional dimensions
/// \defgroup ConversionBetween2and3dimGroup 2d <-> 3d conversion
/// \ingroup MathGroup
/// \{

template <typename T>
[[nodiscard]] inline Vector3<T> to3dim( const Vector2<T> & v )
{
    return { v.x, v.y, T{0} };
}

template <typename T>
[[nodiscard]] inline Vector2<T> to2dim( const Vector3<T> & v )
{
    return { v.x, v.y };
}

template <typename T>
[[nodiscard]] inline Matrix3<T> to3dim( const Matrix2<T> & m )
{
    return { to3dim( m.x ), to3dim( m.y ), Vector3<T>::plusZ() };
}

template <typename T>
[[nodiscard]] inline Matrix2<T> to2dim( const Matrix3<T> & m )
{
    return { to2dim( m.x ), to2dim( m.y ) };
}

template <typename T>
[[nodiscard]] inline AffineXf3<T> to3dim( const AffineXf2<T> & xf )
{
    return { to3dim( xf.A ), to3dim( xf.b ) };
}

template <typename T>
[[nodiscard]] inline AffineXf2<T> to2dim( const AffineXf3<T> & xf )
{
    return { to2dim( xf.A ), to2dim( xf.b ) };
}

/// \}

} // namespace MR
