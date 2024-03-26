#pragma once

#include "MRMesh/MRMeshFwd.h"

#include <cassert>

namespace MR
{

// Common traits for (mathematical) vectors.

template <typename T>
struct VectorTraits
{
    // The base template handles scalars (or just non-vectors).

    using BaseType = T;
    static constexpr int size = 1;

    // Currently this doesn't forward the value, for simplicity. (In all specializations.)
    // For scalars this intentionally doesn't check the index.
    template <typename U>
    [[nodiscard]] static constexpr auto&& getElem( int i, U&& value ) { (void)i; return value; }
};

template <typename T>
struct VectorTraits<Vector2<T>>
{
    using BaseType = T;
    static constexpr int size = 2;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }
};
template <typename T>
struct VectorTraits<Vector3<T>>
{
    using BaseType = T;
    static constexpr int size = 3;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }
};
template <typename T>
struct VectorTraits<Vector4<T>>
{
    using BaseType = T;
    static constexpr int size = 4;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }
};

}
