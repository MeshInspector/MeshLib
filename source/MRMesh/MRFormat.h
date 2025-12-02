#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRFmt.h"

namespace MR
{

template <typename T>
std::string format_as( const Vector2<T>& vec )
{
    return fmt::format( "{} {}", vec.x, vec.y );
}

template <typename T>
std::string format_as( const Vector3<T>& vec )
{
    return fmt::format( "{} {} {}", vec.x, vec.y, vec.z );
}

template <typename T>
std::string format_as( const Vector4<T>& vec )
{
    return fmt::format( "{} {} {} {}", vec.x, vec.y, vec.z, vec.w );
}

template <typename T>
std::string format_as( const Matrix3<T>& mat )
{
    return fmt::format( "{}\n{}\n{}\n", mat.x, mat.y, mat.z );
}

template <typename T>
std::string format_as( const Matrix4<T>& mat )
{
    return fmt::format( "{}\n{}\n{}\n{}\n", mat.x, mat.y, mat.z, mat.w );
}

template <typename T>
std::string format_as( const Plane3<T>& pl )
{
    return fmt::format( "{}\n{}", pl.n, pl.d );
}

template <typename T>
std::string format_as( const Line3<T>& l )
{
    return fmt::format( "{}\n{}", l.n, l.d );
}

template <typename T>
std::string format_as( const TriPoint<T>& tp )
{
    return fmt::format( "{} {}", tp.a, tp.b );
}

template <typename T>
std::string format_as( const AffineXf3<T>& xf )
{
    return fmt::format( "{} {}", xf.A, xf.b );
}

template <typename V>
std::string format_as( const Box<V>& box )
{
    return fmt::format( "{}\n{}", box.min, box.max );
}

MRMESH_API std::string format_as( const BitSet& bs );

} // namespace MR
