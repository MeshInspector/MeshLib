#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRFmt.h"

template <typename T>
struct fmt::formatter<MR::Vector2<T>> : fmt::formatter<T>
{
    template <typename Context>
    constexpr auto format( const MR::Vector2<T>& v, Context& ctx ) const
    {
        using Base = fmt::formatter<T>;
        auto out = ctx.out();
        out = Base::format( v.x, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.y, ctx );
        return out;
    }
};

template <typename T>
struct fmt::formatter<MR::Vector3<T>> : fmt::formatter<T>
{
    template <typename Context>
    constexpr auto format( const MR::Vector3<T>& v, Context& ctx ) const
    {
        using Base = fmt::formatter<T>;
        auto out = ctx.out();
        out = Base::format( v.x, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.y, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.z, ctx );
        return out;
    }
};

template <typename T>
struct fmt::formatter<MR::Vector4<T>> : fmt::formatter<T>
{
    template <typename Context>
    constexpr auto format( const MR::Vector4<T>& v, Context& ctx ) const
    {
        using Base = fmt::formatter<T>;
        auto out = ctx.out();
        out = Base::format( v.x, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.y, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.z, ctx );
        *out++ = ',';
        *out++ = ' ';
        out = Base::format( v.w, ctx );
        return out;
    }
};

template <typename T>
struct fmt::formatter<MR::Matrix3<T>> : fmt::formatter<MR::Vector3<T>>
{
    template <typename Context>
    static constexpr auto formatWith( const MR::Matrix3<T>& mat, Context& ctx, const fmt::formatter<MR::Vector3<T>>& v )
    {
        auto out = ctx.out();
        *out++ = '{';
        out = v.format( mat.x, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = v.format( mat.y, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = v.format( mat.z, ctx );
        *out++ = '}';
        return out;
    }

    template <typename Context>
    constexpr auto format( const MR::Matrix3<T>& mat, Context& ctx ) const
    {
        return formatWith( mat, ctx, *this );
    }
};

template <typename T>
struct fmt::formatter<MR::Matrix4<T>> : fmt::formatter<MR::Vector4<T>>
{
    template <typename Context>
    constexpr auto format( const MR::Matrix4<T>& mat, Context& ctx ) const
    {
        using Vec = fmt::formatter<MR::Vector4<T>>;
        auto out = ctx.out();
        *out++ = '{';
        out = Vec::format( mat.x, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = Vec::format( mat.y, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = Vec::format( mat.z, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = Vec::format( mat.w, ctx );
        *out++ = '}';
        return out;
    }
};

template <typename T>
struct fmt::formatter<MR::AffineXf3<T>> : fmt::formatter<MR::Vector3<T>>
{
    template <typename Context>
    constexpr auto format( const MR::AffineXf3<T>& xf, Context& ctx ) const
    {
        using Mat = fmt::formatter<MR::Matrix3<T>>;
        using Vec = fmt::formatter<MR::Vector3<T>>;
        auto out = ctx.out();
        *out++ = '{';
        out = Mat::formatWith( xf.A, ctx, *this );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = Vec::format( xf.b, ctx );
        *out++ = '}';
        return out;
    }
};

template <typename V>
struct fmt::formatter<MR::Box<V>> : fmt::formatter<V>
{
    template <typename Context>
    constexpr auto format( const MR::Box<V>& box, Context& ctx ) const
    {
        using Vec = fmt::formatter<V>;
        auto out = ctx.out();
        *out++ = '{';
        out = Vec::format( box.min, ctx );
        *out++ = '}';
        *out++ = ',';
        *out++ = ' ';
        *out++ = '{';
        out = Vec::format( box.max, ctx );
        *out++ = '}';
        return out;
    }
};

template <>
struct fmt::formatter<MR::BitSet> : fmt::formatter<std::string>
{
    template <typename Context>
    constexpr auto format( const MR::BitSet& bs, Context& ctx ) const
    {
        using Base = fmt::formatter<std::string>;
        return Base::format( toString( bs ), ctx );
    }

private:
    MRMESH_API static std::string toString( const MR::BitSet& bs );
};
