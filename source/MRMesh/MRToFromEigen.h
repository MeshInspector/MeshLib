#pragma once

#include "MRQuaternion.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRSymMatrix2.h"
#include "MRSymMatrix3.h"

// unknown pragmas
#pragma warning(disable:4068)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#include <Eigen/Core>
#pragma clang diagnostic pop

namespace MR
{

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
static Eigen::Matrix<T, 2, 2> toEigen( const SymMatrix2<T> & m )
{
    Eigen::Matrix<T, 2, 2> res;
    res << m.xx, m.xy,
           m.xy, m.yy;
    return res;
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

} //namespace MR
