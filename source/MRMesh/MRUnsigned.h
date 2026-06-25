#pragma once

#include <cstddef>
#include <type_traits>

// Additional support for unsigned types.

namespace MR::Unsigned
{

// Only callable on unsigned types. Returns the value unchanged.
// The intended usage of this in templates is like this:
//     using std::abs;
//     using Unsigned::abs; // Or `using namespace Unsigned;`.
//     abs(x);
// Doing it this way ensures that you can handle both signed and unsigned standard types, and additionally any classes that provide `abs()` via ADL.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, std::nullptr_t> = nullptr>
[[nodiscard]] constexpr T abs( T value )
{
    return value;
}

}
