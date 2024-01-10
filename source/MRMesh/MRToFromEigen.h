#pragma once

#include "MRQuaternion.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRSymMatrix2.h"
#include "MRSymMatrix3.h"
#include "MRSymMatrix4.h"
#include "MRMatrix2.h"
#include "MRMatrix3.h"
#include "MRMatrix4.h"
#include <MRPch/MREigenCore.h>

namespace MR
{

/// \ingroup MathGroup
/// \{

template <typename T>
[[nodiscard]] inline Vector2<T> fromEigen( const Eigen::Matrix<T, 2, 1> & ev )
{
    return Vector2<T>{ ev.x(), ev.y() };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 2, 1> toEigen( const Vector2<T> & v )
{
    return Eigen::Matrix<T, 2, 1>{ v.x, v.y };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 2, 2> toEigen( const SymMatrix2<T> & m )
{
    Eigen::Matrix<T, 2, 2> res;
    res << m.xx, m.xy,
           m.xy, m.yy;
    return res;
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 2, 2> toEigen( const Matrix2<T> & m )
{
    Eigen::Matrix<T, 2, 2> res;
    res << m.x.x, m.x.y,
           m.y.x, m.y.y;
    return res;
}

template <typename T>
[[nodiscard]] inline Matrix2<T> fromEigen( const Eigen::Matrix<T, 2, 2> & m )
{
    return Matrix2<T> {
        { m( 0, 0 ), m( 0, 1 ) },
        { m( 1, 0 ), m( 1, 1 ) }
    };
}

template <typename T>
[[nodiscard]] inline Vector3<T> fromEigen( const Eigen::Matrix<T, 3, 1> & ev )
{
    return Vector3<T>{ ev.x(), ev.y(), ev.z() };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 3, 1> toEigen( const Vector3<T> & v )
{
    return Eigen::Matrix<T, 3, 1>{ v.x, v.y, v.z };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 3, 3> toEigen( const SymMatrix3<T> & m )
{
    Eigen::Matrix<T, 3, 3> res;
    res << m.xx, m.xy, m.xz,
           m.xy, m.yy, m.yz,
           m.xz, m.yz, m.zz;
    return res;
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 3, 3> toEigen( const Matrix3<T> & m )
{
    Eigen::Matrix<T, 3, 3> res;
    res << m.x.x, m.x.y, m.x.z,
           m.y.x, m.y.y, m.y.z,
           m.z.x, m.z.y, m.z.z;
    return res;
}

template <typename T>
[[nodiscard]] inline Matrix3<T> fromEigen( const Eigen::Matrix<T, 3, 3> & m )
{
    return Matrix3<T> {
        { m( 0, 0 ), m( 0, 1 ), m( 0, 2 ) },
        { m( 1, 0 ), m( 1, 1 ), m( 1, 2 ) },
        { m( 2, 0 ), m( 2, 1 ), m( 2, 2 ) }
    };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 4, 4> toEigen( const SymMatrix4<T> & m )
{
    Eigen::Matrix<T, 4, 4> res;
    res << m.xx, m.xy, m.xz, m.xw,
           m.xy, m.yy, m.yz, m.yw,
           m.xz, m.yz, m.zz, m.zw,
           m.xw, m.yw, m.zw, m.ww;
    return res;
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 4, 4> toEigen( const Matrix4<T> & m )
{
    Eigen::Matrix<T, 4, 4> res;
    res << m.x.x, m.x.y, m.x.z, m.x.w,
           m.y.x, m.y.y, m.y.z, m.y.w,
           m.z.x, m.z.y, m.z.z, m.x.w,
           m.w.x, m.w.y, m.w.z, m.w.w;
    return res;
}

template <typename T>
[[nodiscard]] inline Matrix4<T> fromEigen( const Eigen::Matrix<T, 4, 4> & m )
{
    return Matrix4<T> {
        { m( 0, 0 ), m( 0, 1 ), m( 0, 2 ), m( 0, 3 ) },
        { m( 1, 0 ), m( 1, 1 ), m( 1, 2 ), m( 1, 3 ) },
        { m( 2, 0 ), m( 2, 1 ), m( 2, 2 ), m( 2, 3 ) },
        { m( 3, 0 ), m( 3, 1 ), m( 3, 2 ), m( 3, 3 ) }
    };
}

template <typename T>
[[nodiscard]] inline Vector4<T> fromEigen( const Eigen::Matrix<T, 4, 1> & ev )
{
    return Vector4<T>{ ev.x(), ev.y(), ev.z(), ev.w() };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 4, 1> toEigen( const Vector4<T> & v )
{
    return Eigen::Matrix<T, 4, 1>{ v.x, v.y, v.z, v.w };
}

template <typename T>
[[nodiscard]] inline Eigen::Matrix<T, 4, 1> toEigen( const Vector3<T> & v, T w )
{
    return Eigen::Matrix<T, 4, 1>{ v.x, v.y, v.z, w };
}

template <typename T>
[[nodiscard]] inline Quaternion<T> fromEigen( const Eigen::Quaternion<T> & eq )
{
    return Quaternion<T>{ eq.w(), eq.x(), eq.y(), eq.z() };
}

template <typename T>
[[nodiscard]] inline Eigen::Quaternion<T> toEigen( const Quaternion<T> & q )
{
    return Eigen::Quaternion<T>{ q.a, q.b, q.c, q.d };
}

/// \}

} // namespace MR
