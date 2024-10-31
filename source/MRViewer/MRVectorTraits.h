#pragma once

#include "MRMesh/MRVectorTraits.h"

struct ImVec2;
struct ImVec4;

namespace MR
{

template <>
struct VectorTraits<ImVec2>
{
    using BaseType = float;
    static constexpr int size = 2;

    // Can't change the element type...
    template <std::same_as<float>>
    using ChangeBaseType = ImVec2;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }

    template <typename U = ImVec2> // Adding a template parameter to avoid including the whole `imgui.h`.
    static constexpr U diagonal( float v ) { return U( v, v ); }
};

template <>
struct VectorTraits<ImVec4>
{
    using BaseType = float;
    static constexpr int size = 4;

    // Can't change the element type...
    template <std::same_as<float>>
    using ChangeBaseType = ImVec4;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }

    template <typename U = ImVec4> // Adding a template parameter to avoid including the whole `imgui.h`.
    static constexpr U diagonal( float v ) { return U( v, v, v, v ); }
};

}
