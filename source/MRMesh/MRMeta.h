#pragma once

#include <memory>

// Metaprogramming helpers.

namespace MR::Meta
{

template <typename T>
struct SharedPtrTraits
{
    static constexpr bool isSharedPtr = false;
    using elemType = T;
};
template <typename T>
struct SharedPtrTraits<std::shared_ptr<T>>
{
    static constexpr bool isSharedPtr = true;
    using elemType = T;
};

}
