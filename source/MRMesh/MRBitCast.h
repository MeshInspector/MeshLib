#pragma once

#include <version>

#if __cpp_lib_bit_cast >= 201806L
#include <bit>
#endif

namespace MR
{

/// reinterprets the bits of (from) as a value of type To, which must be of the same size;
/// std::bit_cast where the standard library provides it, and the intrinsic behind it otherwise
template <typename To, typename From>
[[nodiscard]] constexpr To bit_cast( const From& from ) noexcept
{
#if __cpp_lib_bit_cast >= 201806L
    return std::bit_cast<To>( from );
#else
    return __builtin_bit_cast( To, from );
#endif
}

} //namespace MR
