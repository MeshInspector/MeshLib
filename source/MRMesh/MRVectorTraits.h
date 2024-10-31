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
    static constexpr bool supportNoInit = false;

    // Changes the vector element type. For scalars, replaces the whole type.
    template <typename U>
    using ChangeBaseType = U;

    // Currently this doesn't forward the value, for simplicity. (In all specializations.)
    // For scalars this intentionally doesn't check the index.
    template <typename U>
    [[nodiscard]] static constexpr auto&& getElem( int i, U&& value ) { (void)i; return value; }

    template <typename U = T> // Adding a template parameter to allow instantiating the rest of the class with `T == void`.
    static constexpr U diagonal( U v ) { return v; }
};

template <typename T>
struct VectorTraits<Vector2<T>>
{
    using BaseType = T;
    static constexpr int size = 2;
    static constexpr bool supportNoInit = true;

    template <typename U>
    using ChangeBaseType = Vector2<U>;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }

    static constexpr auto diagonal( T v ) { return Vector2<T>::diagonal( v ); }
};

template <typename T>
struct VectorTraits<Vector3<T>>
{
    using BaseType = T;
    static constexpr int size = 3;
    static constexpr bool supportNoInit = true;

    template <typename U>
    using ChangeBaseType = Vector3<U>;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }

    static constexpr auto diagonal( T v ) { return Vector3<T>::diagonal( v ); }
};

template <typename T>
struct VectorTraits<Vector4<T>>
{
    using BaseType = T;
    static constexpr int size = 4;
    static constexpr bool supportNoInit = true;

    template <typename U>
    using ChangeBaseType = Vector4<U>;

    template <typename U>
    [[nodiscard]] static auto&& getElem( int i, U&& value )
    {
        // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
        // GCC and Clang optimize both in the same manner.
        return ( &value.x )[i];
    }

    static constexpr auto diagonal( T v ) { return Vector4<T>::diagonal( v ); }
};

}
