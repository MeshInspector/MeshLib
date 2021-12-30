#pragma once
#include "MRMeshFwd.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix3.h"
#include "MRMatrix4.h"
#include "MRPlane3.h"
#include "MRBitSet.h"
#include "MRTriPoint.h"
#include "MRAffineXf3.h"
#include "MRPointOnFace.h"

// simple Vector structure elements are separated with spaces ' '
// complex structures (Matrix, Plane, AffineXf, ...) elements (Vector, numerics) are separated with '\n'

namespace MR
{
    // =====================================================================
    // Vector
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Vector2<T>& vec )
    {
        return s << vec.x << ' ' << vec.y;
    }
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Vector3<T>& vec )
    {
        return s << vec.x << ' ' << vec.y << ' ' << vec.z;
    }
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Vector4<T>& vec )
    {
        return s << vec.x << ' ' << vec.y << ' ' << vec.z << ' ' << vec.w;
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Vector2<T>& vec )
    {
        return s >> vec.x >> vec.y;
    }
    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Vector3<T>& vec )
    {
        return s >> vec.x >> vec.y >> vec.z;
    }
    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Vector4<T>& vec )
    {
        return s >> vec.x >> vec.y >> vec.z >> vec.w;
    }

    // =====================================================================
    // Matrix3
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Matrix3<T>& mat )
    {
        return s << mat.x << '\n' << mat.y << '\n' << mat.z << '\n';
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Matrix3<T>& mat )
    {
        return s >> mat.x >> mat.y >> mat.z;
    }

    // =====================================================================
    // Matrix4
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Matrix4<T>& mat )
    {
        return s << mat.x << '\n' << mat.y << '\n' << mat.z << '\n' << mat.w << '\n';
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Matrix4<T>& mat )
    {
        return s >> mat.x >> mat.y >> mat.z >> mat.w;
    }

    // =====================================================================
    // Plane3
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Plane3<T>& pl )
    {
        return s << pl.n << '\n' << pl.d;
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Plane3<T>& pl )
    {
        return s >> pl.n >> pl.d;
    }

    // =====================================================================
    // Line3
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const Line3<T>& l )
    {
        return s << l.p << '\n' << l.d;
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, Line3<T>& l )
    {
        return s >> l.p >> l.d;
    }

    // =====================================================================
    // TriPoint
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const TriPoint<T>& tp )
    {
        return s << tp.a << ' ' << tp.b;
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, TriPoint<T>& tp )
    {
        s >> tp.a >> tp.b;
        return s;
    }

    // =====================================================================
    // AffineXf3
    template <typename T>
    inline std::ostream& operator << ( std::ostream& s, const AffineXf3<T>& xf )
    {
        return s << xf.A << xf.b;
    }

    template <typename T>
    inline std::istream& operator >> ( std::istream& s, AffineXf3<T>& xf )
    {
        return s >> xf.A >> xf.b;
    }

    // =====================================================================
    // PointOnFace
    inline std::ostream& operator << ( std::ostream& s, const PointOnFace& pof )
    {
        return s << pof.face << '\n' << pof.point;
    }

    inline std::istream& operator >> ( std::istream& s, PointOnFace& pof )
    {
        int a;
        s >> a >> pof.point;
        pof.face = FaceId( a );
        return s;
    }


    // =====================================================================
    // Box
    template<typename V>
    inline std::ostream& operator << ( std::ostream& s, const Box<V>& box )
    {
        return s << box.min << '\n' << box.max;
    }

    template<typename V>
    inline std::istream& operator >> ( std::istream& s, Box<V>& box )
    {
        return s >> box.min >> box.max;
    }
}
